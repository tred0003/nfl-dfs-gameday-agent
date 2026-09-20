"""Cornerback coverage grades, for pass-catcher matchup context.

DK's `dk_oprk` is a team-level rank against a position. It cannot tell you that
a receiver draws a specific shutdown corner, which is the question a DFS player
actually asks. This module answers the part of it that free data supports.

THREE SOURCES, because no single one is sufficient, and each is used only for
what it is authoritative about:

    is he a CB?      nflverse player ids `position`
    what team NOW?   Sleeper's player endpoint
    how good?        nflverse PFR advanced defense

The id crosswalk carries both `pfr_id` and `sleeper_id`, which is the only
reason these three join at all.

Everything is fetched from nflverse and cached, never read from data/raw/ --
see _load_ids for the incident that forced that. data/raw/ is a frozen
snapshot for training; grading wants current data and, more importantly, wants
the SAME data everywhere it runs.

WHY NOT ONE SOURCE:
- nflverse weekly rosters label every defensive back "DB" in 2026. There is no
  CB granularity, so the position question cannot be answered there.
- Sleeper knows current teams and depth chart order, but splits corners
  arbitrarily between "CB" and "DB": Carlton Davis is CB, Christian Gonzalez is
  DB, and they are on the same team. Filtering Sleeper on "CB" silently drops
  Gonzalez, Joey Porter Jr. and Derek Stingley Jr.
- PFR has the coverage numbers but its team column is the team a player played
  FOR IN THAT WEEK, so it goes stale across an offseason.
- depth_charts.parquet stops at 2024.

WHAT THIS CANNOT DO: say whether a corner shadows. Knowing that a receiver
draws a particular defender needs per-snap alignment data, which no free source
publishes. `role` below is inferred from average depth of target -- a boundary
corner is thrown at deeper than a slot corner -- and that is a tendency, not an
assignment. Any tool built on this must say so rather than implying a matchup
is locked in.

SAMPLE SIZE: nflverse publishes defensive stats in batches days after games,
so early in a season a grade is mostly the PRIOR season. That is defensible
for corner talent, which is fairly stable year to year, but it is not "how he
is playing this week" and should not be described as such.
"""

from __future__ import annotations

import argparse

import pandas as pd

SEASONS = (2025, 2026)

# Targets of league-average performance mixed into every player's rate stats.
# A corner with 12 targets and a 40.0 passer rating allowed is not the best
# cover man alive, he is a small sample -- shrinking toward the league mean
# says that without throwing the player out entirely, which a hard minimum
# does. Roughly half a starter's season.
SHRINK_TARGETS = 30

# Below this a grade is too noisy to report at all, shrinkage or not.
MIN_TARGETS = 15

# A corner has to actually be on the field to be graded. Without this the
# board's top spot went to a journeyman with 19 targets in 9 games: the
# avoidance component read his 2.1 targets per game as "quarterbacks refuse to
# test him" when the truth was that he played a fifth of the snaps. Per-game
# volume measures playing time at least as much as reputation, which is why
# the grade below uses targets per hundred coverage snaps instead.
MIN_SNAP_SHARE = 0.50

# Average depth of target, which separates where a corner is thrown at rather
# than where he lines up. Three buckets rather than two because the middle is
# genuinely ambiguous -- forcing a binary labelled Joey Porter Jr. a slot
# corner at 8.05, which he is not.
ADOT_SLOT = 8.0
ADOT_BOUNDARY = 11.0


NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"

# How long a fetched nflverse file stays usable. Defensive stats are published
# in batches days after games, so an hourly refresh would buy nothing.
FETCH_TTL = 6 * 3600


def _cached_fetch(name: str, fetch) -> pd.DataFrame:
    """Fetch a frame and keep it on disk, so grading costs one request a day.

    The cache is an optimisation only: an unwritable directory degrades this
    to fetching every time, never to failing. Agent Platform does not promise
    a writable /app.
    """
    import time

    from sleeper_client import cache_dir

    path = cache_dir() / name
    if path.exists() and time.time() - path.stat().st_mtime < FETCH_TTL:
        try:
            return pd.read_parquet(path)
        except OSError:
            pass

    df = fetch()
    try:
        df.to_parquet(path, index=False)
    except OSError:
        pass
    return df


def _nflverse(release: str, filename: str, seasons: tuple[int, ...]) -> pd.DataFrame:
    """Read a per-season nflverse release asset straight from GitHub.

    Not via nfl_data_py. Its `import_weekly_pfr` rejects s_type="def" in an
    input validator -- "must be one of pass, rec, or rush" -- even though the
    URL it builds supports def perfectly well and the file is published. The
    library is the obstacle here, not the data.

    A season that 404s is skipped rather than fatal: the current season's file
    does not exist until nflverse publishes it.
    """
    frames = []
    for year in seasons:
        fname = filename.format(year)
        try:
            frames.append(_cached_fetch(
                fname, lambda f=fname: pd.read_parquet(f"{NFLVERSE}/{release}/{f}")))
        except Exception:
            continue
    if not frames:
        raise RuntimeError(f"no {release} data available for {seasons}")
    return pd.concat(frames, ignore_index=True)


def _load_pfr_def(seasons: tuple[int, ...]) -> pd.DataFrame:
    """PFR advanced defense, ALWAYS from nflverse -- see _load_ids for why."""
    return _nflverse("pfr_advstats", "advstats_week_def_{}.parquet", seasons)


def _load_ids() -> pd.DataFrame:
    """The player id crosswalk, ALWAYS from nflverse, never the repo's copy.

    ALL THREE LOADERS HERE IGNORE data/raw/ ON PURPOSE.

    They used to prefer the local parquets and fall back to the network, which
    made the same code produce different grades in different places. The repo's
    2026 PFR file holds 50 rows across 2 games; nflverse serves 485 rows across
    17 games and two weeks. Because a grade is a PERCENTILE within the graded
    pool, more data changes who qualifies and therefore changes everyone's
    number: Carlton Davis came back 83.5 on the laptop and 86.2 deployed on the
    same afternoon, and Max Melton moved 32.0 to 42.6 -- enough to change a
    lineup decision.

    The deployed numbers were the RIGHT ones. The laptop was reading a stale
    snapshot and quietly grading on a fifth of the season's defensive data.

    A grade that depends on where it runs is not a grade, and "prefer local,
    fall back to network" guarantees exactly that whenever local is behind.
    Fetching always, cached for a day, costs one request and removes the whole
    class of problem. data/raw/ stays as it is for training, which wants a
    frozen snapshot for the opposite reason.
    """
    return _cached_fetch("nflverse_ids.parquet", lambda: _import_ids())


def _import_ids() -> pd.DataFrame:
    import nfl_data_py as nfl
    return nfl.import_ids()


def _load_snaps(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Snap counts, ALWAYS from nflverse -- see _load_ids for why."""
    return _nflverse("snap_counts", "snap_counts_{}.parquet", seasons)


def _passer_rating(cmp_pct: pd.Series, ypt: pd.Series,
                   td_rate: pd.Series, int_rate: pd.Series) -> pd.Series:
    """NFL passer rating, from the four component rates, each capped at 2.375."""
    a = ((cmp_pct - 0.30) * 5).clip(0, 2.375)
    b = ((ypt - 3) * 0.25).clip(0, 2.375)
    c = (td_rate * 20).clip(0, 2.375)
    d = (2.375 - int_rate * 25).clip(0, 2.375)
    return (a + b + c + d) / 6 * 100


def _shrink(rate: pd.Series, n: pd.Series, league: float) -> pd.Series:
    """Pull a rate toward the league mean in proportion to how little data backs it."""
    return (n * rate + SHRINK_TARGETS * league) / (n + SHRINK_TARGETS)


def cb_grades(seasons: tuple[int, ...] = SEASONS) -> pd.DataFrame:
    """One row per cornerback: coverage grade, role, volume and current team.

    `coverage_grade` runs 0-100, higher meaning harder to throw on. It is a
    weighted blend of percentile ranks within the qualified corner pool:

        45%  passer rating allowed        (the standard single-number metric)
        35%  yards allowed per target     (less touchdown-noisy than rating)
        20%  targets per 100 cover snaps  (avoidance)

    Avoidance is per SNAP, not per game. Per game measures playing time as
    much as reputation, and graded backups accordingly -- see MIN_SNAP_SHARE.

    It is still weighted lowest because it stays confounded even per snap: a
    corner sees fewer throws when his pass rush is good, when his team is
    ahead, and when the corner opposite him is worse. It is real evidence that
    quarterbacks decline to test him, but it is not purely his doing.

    Only corners playing at least MIN_SNAP_SHARE of their defense's snaps are
    graded. Everyone else is returned with coverage_grade None rather than
    dropped, so a caller can still see that a name exists and is a backup.
    """
    d = _load_pfr_def(seasons)
    ids = _load_ids()[["pfr_id", "sleeper_id", "position", "name"]]
    ids = ids.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    d = d.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="left")
    d = d[d["position"].eq("CB")]

    g = d.groupby(["pfr_player_id", "pfr_player_name"], as_index=False).agg(
        games=("game_id", "nunique"),
        targets=("def_targets", "sum"),
        completions=("def_completions_allowed", "sum"),
        yards=("def_yards_allowed", "sum"),
        tds=("def_receiving_td_allowed", "sum"),
        ints=("def_ints", "sum"),
        adot=("def_adot", "mean"),
        sleeper_id=("sleeper_id", "last"),
    )
    g = g[g["targets"] >= MIN_TARGETS].copy()
    if g.empty:
        return g

    # Coverage snaps, the honest denominator for "is he avoided".
    snaps = _load_snaps(seasons)
    snaps = snaps[snaps["season"].isin(seasons)]
    sg = snaps.groupby("pfr_player_id", as_index=False).agg(
        cover_snaps=("defense_snaps", "sum"),
        snap_share=("defense_pct", "mean"),
    )
    g = g.merge(sg, left_on="pfr_player_id", right_on="pfr_player_id", how="left")
    g["snap_share"] = g["snap_share"].round(3)

    # League baselines, computed on pooled totals rather than an average of
    # per-player rates, so a low-volume corner does not swing the baseline.
    lg_cmp = g["completions"].sum() / g["targets"].sum()
    lg_ypt = g["yards"].sum() / g["targets"].sum()
    lg_td = g["tds"].sum() / g["targets"].sum()
    lg_int = g["ints"].sum() / g["targets"].sum()

    cmp_pct = _shrink(g["completions"] / g["targets"], g["targets"], lg_cmp)
    ypt = _shrink(g["yards"] / g["targets"], g["targets"], lg_ypt)
    td_rate = _shrink(g["tds"] / g["targets"], g["targets"], lg_td)
    int_rate = _shrink(g["ints"] / g["targets"], g["targets"], lg_int)

    g["completion_pct_allowed"] = (cmp_pct * 100).round(1)
    g["yards_per_target"] = ypt.round(2)
    g["passer_rating_allowed"] = _passer_rating(cmp_pct, ypt, td_rate, int_rate).round(1)
    g["targets_per_game"] = (g["targets"] / g["games"]).round(2)
    g["targets_per_100_snaps"] = (g["targets"] / g["cover_snaps"] * 100).round(2)

    # Only real starters get a grade. Percentiles are computed within THAT
    # pool, so a backup cannot dilute the scale he is not measured against.
    starter = g["snap_share"].ge(MIN_SNAP_SHARE) & g["cover_snaps"].notna()
    q = g[starter]

    p_rating = q["passer_rating_allowed"].rank(pct=True, ascending=False)
    p_ypt = q["yards_per_target"].rank(pct=True, ascending=False)
    p_avoid = q["targets_per_100_snaps"].rank(pct=True, ascending=False)
    g.loc[starter, "coverage_grade"] = (
        (0.45 * p_rating + 0.35 * p_ypt + 0.20 * p_avoid) * 100
    ).round(1)
    g["is_full_time"] = starter

    def _role(a):
        if pd.isna(a):
            return None
        if a < ADOT_SLOT:
            return "slot"
        return "boundary" if a > ADOT_BOUNDARY else "mixed"

    g["role"] = g["adot"].apply(_role)
    g["adot"] = g["adot"].round(2)
    return g.sort_values("coverage_grade", ascending=False,
                         na_position="last").reset_index(drop=True)


def cb_board() -> pd.DataFrame:
    """Coverage grades joined to CURRENT team, depth chart slot and injury.

    PFR's team column is where a player lined up in a given week, so it is a
    season out of date the moment free agency starts. Sleeper is consulted for
    where each corner actually plays now, joined on the sleeper_id that
    ids.parquet supplies.

    A corner with no current Sleeper row is dropped: he is retired, a free
    agent, or otherwise not lining up against anybody this week.
    """
    from sleeper_client import fetch_players

    g = cb_grades()
    sleeper = fetch_players()[
        ["player_id", "team", "injury_status", "depth_chart_order", "name"]
    ].rename(columns={"name": "current_name"})

    g["sleeper_id"] = g["sleeper_id"].astype("string").str.replace(
        r"\.0$", "", regex=True
    )
    out = g.merge(sleeper, left_on="sleeper_id", right_on="player_id", how="inner")
    return out.sort_values("coverage_grade", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cornerback coverage grades")
    p.add_argument("--team", help="show one team's corners")
    p.add_argument("--top", type=int, default=15)
    a = p.parse_args()

    b = cb_board()
    cols = ["pfr_player_name", "team", "coverage_grade", "passer_rating_allowed",
            "yards_per_target", "targets_per_100_snaps", "snap_share", "role",
            "targets", "injury_status"]
    if a.team:
        b = b[b["team"].eq(a.team.upper())]
        print(f"{a.team.upper()} cornerbacks ({len(b)}):")
        print(b[cols].to_string(index=False))
    else:
        print(f"Top {a.top} cornerbacks by coverage grade "
              f"(of {len(b)} with {MIN_TARGETS}+ targets):")
        print(b.head(a.top)[cols].to_string(index=False))

"""Pull historical NFL data (features + labels) from nflverse into data/raw/.

Free, no API key. This is where the LABEL comes from: what each player
actually scored, week by week.

Run: python src/fetch_nflverse.py --start 2016 --end 2025
Extras only (play-by-play, participation, charting...):
     python src/fetch_nflverse.py --extras-only --end 2026
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd
import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"


def _safe_parquet(df: pd.DataFrame, path: Path, label: str) -> None:
    """nflverse mixes types inside object columns; parquet refuses those."""
    for col in df.select_dtypes(include="object").columns:
        if df[col].dropna().map(type).nunique() > 1:
            df[col] = df[col].astype(str).where(df[col].notna(), None)
    # Write beside the target and swap, rather than writing through it. A pull
    # interrupted halfway -- dropped connection, Ctrl-C -- otherwise leaves a
    # truncated parquet where a good one used to be, and the next reader gets a
    # confusing schema error instead of the previous file.
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, engine="pyarrow", index=False)
    os.replace(tmp, path)
    print(f"  {label:<12} {str(df.shape):>16}  -> {path.name}")


def _pull(fn, years: list[int], path: Path, label: str) -> None:
    """Fetch `years`, trimming the tail until upstream actually has the data.

    nflverse publishes each dataset on its own schedule -- as of Sept 2026,
    schedules/snaps/injuries have 2025 but weekly player stats do not. Trimming
    beats hard-failing the whole pull over one lagging file.
    """
    yrs = list(years)
    while yrs:
        try:
            _safe_parquet(fn(yrs), path, label)
            if yrs[-1] != years[-1]:
                print(f"    ^ upstream has {label} only through {yrs[-1]}")
            return
        except Exception:
            yrs = yrs[:-1]
    print(f"  {label:<12} UNAVAILABLE for {years[0]}-{years[-1]}")


def pull_player_stats(years: list[int], path: Path) -> None:
    """Weekly player stats from nflverse's current `stats_player` release.

    nfl_data_py.import_weekly_data reads the retired `player_stats` release,
    which stops at 2024. The new release has the same box-score numbers
    (checked on 2024: targets, yards, carries, receptions agree 99.9-100%) and
    runs through the current season. One real difference: target_share,
    air_yards_share and wopr are 0 for a zero-target game instead of blank.
    Columns are renamed to the old names so nothing downstream changes.
    """
    _safe_parquet(_load_player_stats(years), path, "weekly")


def _load_player_stats(years: list[int]) -> pd.DataFrame:
    """The fetch half of pull_player_stats, so update() can reuse it."""
    frames = []
    for y in years:
        r = requests.get(f"{RELEASES}/stats_player/stats_player_week_{y}.parquet", timeout=120)
        if r.status_code == 404:
            print(f"    ^ stats_player {y} not published")
            continue
        r.raise_for_status()
        frames.append(pd.read_parquet(io.BytesIO(r.content)))
    df = pd.concat(frames, ignore_index=True).rename(columns={
        "team": "recent_team",
        "passing_interceptions": "interceptions",
        "sacks_suffered": "sacks",
        "sack_yards_lost": "sack_yards",
    })
    return df


def fetch(start: int, end: int) -> None:
    years = list(range(start, end + 1))
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"pulling {start}-{end}")

    # weekly per-player box score -- the label lives here
    pull_player_stats(years, RAW / "weekly.parquet")

    # game context: Vegas lines, weather, roof, rest days
    _pull(nfl.import_schedules, years, RAW / "schedules.parquet", "schedules")

    # snap share -- the cleanest "is he actually on the field" signal
    _pull(nfl.import_snap_counts, years, RAW / "snaps.parquet", "snaps")

    # weekly injury report (practice participation + game status)
    _pull(nfl.import_injuries, years, RAW / "injuries.parquet", "injuries")

    # id crosswalk: DK names -> nflverse player ids
    _safe_parquet(nfl.import_ids(), RAW / "ids.parquet", "ids")

    # depth charts -- starter vs backup, and who moves up when someone is out
    _pull(nfl.import_depth_charts, years, RAW / "depth_charts.parquet", "depth")


# ------------------------------------------------------------------ extras
def _download(tag: str, filename: str, dest: Path, refresh: bool) -> str:
    """Save one nflverse release file as-is. Returns 'ok', 'kept' or 'missing'.

    These are pulled straight from the release page rather than through
    nfl_data_py, which lags behind upstream and has no loader for several of
    them. Files are kept per year: play-by-play alone is ~20 MB a season.
    """
    if dest.exists() and not refresh:
        return "kept"
    r = requests.get(f"{RELEASES}/{tag}/{filename}", timeout=120)
    if r.status_code == 404:
        return "missing"
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return "ok"


def _pull_release(tag: str, pattern: str, years: list[int] | None, folder: str,
                  label: str, refresh: bool) -> None:
    names = [pattern.format(year=y) for y in years] if years else [pattern]
    status = {n: _download(tag, n, RAW / folder / n, refresh) for n in names}
    got = [n for n, s in status.items() if s != "missing"]
    missing = [n for n, s in status.items() if s == "missing"]
    mb = sum((RAW / folder / n).stat().st_size for n in got) / 1e6
    print(f"  {label:<14} {len(got):>2} files, {mb:6.1f} MB  -> {folder}/")
    if missing:
        print(f"    ^ not published upstream: {', '.join(missing)}")


def fetch_extras(start: int, end: int, refresh: bool) -> None:
    """Play-level and advanced data. Re-run with --refresh during a season:
    the current year's files keep growing, earlier years don't change."""
    yrs = lambda first: list(range(max(start, first), end + 1))  # noqa: E731
    print(f"pulling extras {start}-{end}")

    # every play: down, distance, field position, air yards, EPA.
    # Source for red-zone usage, pace, pass rate, and rebuilding 2025 weekly.
    _pull_release("pbp", "play_by_play_{year}.parquet", yrs(1999),
                  "pbp", "play-by-play", refresh)

    # all 22 player ids on the field for each play -> weekly routes-run proxy
    _pull_release("pbp_participation", "pbp_participation_{year}.parquet", yrs(2016),
                  "participation", "participation", refresh)

    # human film charting: catchable, contested, drops, interception-worthy
    _pull_release("ftn_charting", "ftn_charting_{year}.parquet", yrs(2022),
                  "ftn", "ftn charting", refresh)

    # Pro Football Reference weekly: broken tackles, yards after contact, drops
    for kind in ("rush", "rec", "pass", "def"):
        _pull_release("pfr_advstats", f"advstats_week_{kind}_{{year}}.parquet", yrs(2018),
                      "pfr", f"pfr {kind}", refresh)

    # Next Gen Stats weekly: separation, cushion, YAC over expected
    for kind in ("receiving", "rushing", "passing"):
        _pull_release("nextgen_stats", f"ngs_{kind}.parquet", None,
                      "ngs", f"ngs {kind}", refresh)

    # one file each, all years
    _pull_release("draft_picks", "draft_picks.parquet", None, "players", "draft picks", refresh)
    _pull_release("combine", "combine.parquet", None, "players", "combine", refresh)
    _pull_release("contracts", "historical_contracts.parquet", None, "players", "contracts", refresh)


# ------------------------------------------------------------------ in-season update
# A full pull rewrites eleven seasons to refresh one week. That is slow enough
# that you skip it, and skipping it is how a lineup gets built on a stale injury
# report -- which pre-flight then correctly refuses, leaving you stuck between a
# check you cannot satisfy and a download you will not run.
#
# Worse, the obvious shortcut is destructive. `_pull` writes whatever it fetched
# over the whole file, so `--start 2026 --end 2026` does not add 2026, it
# REPLACES 2016-2025 with it. Nothing warns. The file just gets smaller.
#
# So: update one season, merged into what is already there.
#
# Only the tables that actually move during a season are listed. `ids` is a name
# crosswalk and barely changes; history never changes at all.

UPDATABLE = {
    # label:      (loader, filename, what "fresh" means for this one)
    "injuries":   (lambda yrs: nfl.import_injuries(yrs), "injuries.parquet",
                   "a new practice report daily; inactives 90 min before kickoff"),
    "schedules":  (lambda yrs: nfl.import_schedules(yrs), "schedules.parquet",
                   "where the Vegas lines live -- they move all week"),
    "snaps":      (lambda yrs: nfl.import_snap_counts(yrs), "snaps.parquet",
                   "after each week's games"),
    "weekly":     (_load_player_stats, "weekly.parquet",
                   "after each week's games -- this is the label"),
    "depth":      (lambda yrs: nfl.import_depth_charts(yrs), "depth_charts.parquet",
                   "teams reshuffle weekly"),
}


def _season_slice(df: pd.DataFrame, season: int) -> pd.Series:
    """Rows for one season. Dtype-proof: nflverse gives season as int or float."""
    return pd.to_numeric(df["season"], errors="coerce").eq(float(season))


def _freshness(df: pd.DataFrame, label: str) -> str:
    """One line describing how current this slice is, in the terms that matter.

    Max week is the right measure for everything except schedules, which are
    published for all 18 weeks before the season starts. For those the number
    that moves during a week is how many games have a line yet.
    """
    if df.empty:
        return "no rows"
    wk = int(pd.to_numeric(df["week"], errors="coerce").max())
    if label == "schedules" and "total_line" in df.columns:
        priced = int(df["total_line"].notna().sum())
        return f"wk{wk}, {priced}/{len(df)} games priced"
    return f"wk{wk}, {len(df):,} rows"


def update(season: int, only: list[str] | None = None) -> None:
    """Refresh one season in place. Merges; never overwrites other seasons."""
    print(f"updating {season} in {RAW}")
    print("  (history untouched -- this replaces only this season's rows)")
    print("")

    for label, (loader, fname, why) in UPDATABLE.items():
        if only and label not in only:
            continue
        path = RAW / fname
        if not path.exists():
            print(f"  {label:<10} SKIP -- {fname} does not exist. Run a full pull first.")
            continue

        old = pd.read_parquet(path)
        old_slice = old[_season_slice(old, season)]
        before = _freshness(old_slice, label)

        try:
            new = loader([season])
        except Exception as e:
            print(f"  {label:<10} FAILED to fetch: {type(e).__name__}: {str(e)[:70]}")
            print(f"             nothing written. {before} still on disk.")
            continue

        if new is None or new.empty:
            print(f"  {label:<10} upstream has nothing for {season}. Unchanged ({before}).")
            continue

        # REFUSE TO GO BACKWARDS. A partial publish, a truncated response or an
        # upstream mid-update all look like a successful download of less data.
        # Without this the good file is quietly replaced by a worse one and the
        # only symptom is a pre-flight failure next week that makes no sense.
        old_wk = pd.to_numeric(old_slice["week"], errors="coerce").max() if len(old_slice) else -1
        new_wk = pd.to_numeric(new["week"], errors="coerce").max()
        if len(old_slice) and (new_wk < old_wk or len(new) < len(old_slice) * 0.9):
            print(f"  {label:<10} REFUSED -- download is smaller than what you have")
            print(f"             on disk {before}  ->  offered wk{int(new_wk)}, {len(new):,} rows")
            print(f"             nothing written. Try again later.")
            continue

        # Schema drift: keep the column set downstream already expects, and say
        # so rather than silently adding or dropping features mid-season.
        added = [c for c in new.columns if c not in old.columns]
        gone = [c for c in old.columns if c not in new.columns]
        if added:
            print(f"  {label:<10} note: upstream added {len(added)} column(s), not adopted: {added[:4]}")
        if gone:
            print(f"  {label:<10} WARN: upstream dropped {len(gone)} column(s) this file has: {gone[:4]}")
            print(f"             they will be null for {season}.")
        new = new.reindex(columns=old.columns)

        merged = pd.concat([old[~_season_slice(old, season)], new], ignore_index=True)
        _safe_parquet(merged, path, label)
        after = _freshness(merged[_season_slice(merged, season)], label)
        arrow = "unchanged" if before == after else f"{before}  ->  {after}"
        print(f"             {arrow}")

    print("")
    print("Unchanged is often correct: snaps and box scores cannot advance until")
    print("the games are played. Injuries and lines are the ones that move midweek.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2016)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--extras-only", action="store_true",
                   help="skip the core tables the model is already built on")
    p.add_argument("--refresh", action="store_true",
                   help="re-download extras that already exist")
    p.add_argument("--update", type=int, metavar="SEASON",
                   help="in-season refresh: re-pull ONLY this season and merge it into "
                        "the existing files. Seconds rather than minutes, and it cannot "
                        "delete history the way --start SEASON would. Run it before a "
                        "slate for injuries and lines, and after the games for snaps "
                        "and box scores.")
    p.add_argument("--only", nargs="+", metavar="TABLE",
                   help=f"with --update, limit to these tables: {', '.join(UPDATABLE)}")
    a = p.parse_args()

    if a.update:
        update(a.update, a.only)
    else:
        if not a.extras_only:
            fetch(a.start, a.end)
        fetch_extras(a.start, a.end, a.refresh)

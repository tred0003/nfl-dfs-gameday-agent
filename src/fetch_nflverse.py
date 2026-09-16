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
    df.to_parquet(path, engine="pyarrow", index=False)
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
    _safe_parquet(df, path, "weekly")


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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2016)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--extras-only", action="store_true",
                   help="skip the core tables the model is already built on")
    p.add_argument("--refresh", action="store_true",
                   help="re-download extras that already exist")
    a = p.parse_args()
    if not a.extras_only:
        fetch(a.start, a.end)
    fetch_extras(a.start, a.end, a.refresh)

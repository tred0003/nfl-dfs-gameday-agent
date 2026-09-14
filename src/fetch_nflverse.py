"""Pull historical NFL data (features + labels) from nflverse into data/raw/.

Free, no API key. This is where the LABEL comes from: what each player
actually scored, week by week.

Run: python src/fetch_nflverse.py --start 2016 --end 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


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


def fetch(start: int, end: int) -> None:
    years = list(range(start, end + 1))
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"pulling {start}-{end}")

    # weekly per-player box score -- the label lives here
    _pull(nfl.import_weekly_data, years, RAW / "weekly.parquet", "weekly")

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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2016)
    p.add_argument("--end", type=int, default=2025)
    fetch(**vars(p.parse_args()))

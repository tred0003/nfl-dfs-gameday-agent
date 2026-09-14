"""Backfill historical DraftKings NFL salaries from the RotoGuru archive.

Why this exists: the live DK endpoint only exposes the CURRENT slate. Testing
any "buy low" / value hypothesis needs price history, and DK does not publish it.

Coverage is 2016-2021. RotoGuru stopped publishing after the 2021 season
(verified: 2022+ returns an empty table), so seasons after that have to come
from your own weekly snapshots via dk_client.py -- which is exactly why those
snapshots start now.

Handy property: RotoGuru ships salary AND actual DK points in the same row, so
value analysis needs no join to nflverse and no player-name matching.

Run: python src/fetch_salaries.py
"""

from __future__ import annotations

import argparse
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

URL = "http://rotoguru1.com/cgi-bin/fyday.pl?week={wk}&year={yr}&game=dk&scsv=1"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; personal use)"}

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
FIRST_YEAR, LAST_YEAR = 2016, 2021   # RotoGuru coverage
DELAY = 0.4                          # be polite; ~100 requests total


def fetch_week(year: int, week: int) -> pd.DataFrame:
    r = requests.get(URL.format(wk=week, yr=year), headers=HEADERS, timeout=30)
    r.raise_for_status()
    m = re.search(r"<pre>(.*?)</pre>", r.text, re.S | re.I)
    if not m:
        return pd.DataFrame()

    rows = [l for l in m.group(1).split("\n") if l.count(";") >= 8]
    if len(rows) < 2:
        return pd.DataFrame()

    df = pd.read_csv(StringIO("\n".join(rows)), sep=";")
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns={
        "Week": "week", "Year": "season", "GID": "gid", "Name": "name",
        "Pos": "position", "Team": "team", "h/a": "home_away", "Oppt": "opponent",
        "DK points": "dk_points", "DK salary": "salary",
    })


def backfill(first: int, last: int) -> None:
    frames, missing = [], []
    for year in range(first, last + 1):
        got = 0
        for week in range(1, 18):
            try:
                df = fetch_week(year, week)
            except Exception as e:
                print(f"  {year} wk{week:>2}  ERROR {type(e).__name__}")
                continue
            if df.empty:
                missing.append((year, week))
            else:
                frames.append(df)
                got += len(df)
            time.sleep(DELAY)
        print(f"  {year}: {got:,} rows")

    if not frames:
        raise RuntimeError("no salary data returned -- check the archive is still up")

    out = pd.concat(frames, ignore_index=True)
    out["salary"] = pd.to_numeric(out["salary"], errors="coerce")
    out["dk_points"] = pd.to_numeric(out["dk_points"], errors="coerce")
    out = out.dropna(subset=["salary"])          # unpriced players aren't on a slate
    out["position"] = out["position"].str.strip().str.upper()
    out["name"] = out["name"].str.strip()

    path = RAW / "dk_salaries.parquet"
    out.to_parquet(path, index=False)
    print(f"\n{len(out):,} priced player-weeks -> {path.name}")
    print(f"seasons {out.season.min()}-{out.season.max()}, {out.week.nunique()} distinct weeks")
    if missing:
        print(f"empty weeks skipped: {len(missing)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--first", type=int, default=FIRST_YEAR)
    p.add_argument("--last", type=int, default=LAST_YEAR)
    a = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"backfilling DK salaries {a.first}-{a.last}")
    backfill(a.first, a.last)

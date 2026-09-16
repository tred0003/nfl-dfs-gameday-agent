"""Per-position deal boards -- the report you read next to the optimizer's lineup.

The optimizer answers one question exactly: what is the highest-projected legal
lineup under the cap. That is the right answer to that question and it is not
the only thing you want to know before lock, because the optimizer cannot see:

  - a beat report that dropped twenty minutes ago
  - that you do not believe the model on one specific guy
  - that you are entering a tournament and want a player nobody else has

All three end the same way: you override one slot by hand. The board exists to
make that override an informed trade instead of a vibe, by answering "if I plug
this guy in, what does it cost me, and can I even afford him?"

WHAT EACH COLUMN IS FOR

  value      projection minus what the salary implies you should get. The
             "deal" column -- positive means the model likes him more than DK
             priced him. This is what you scan first.

  ceiling    the 85th-percentile model, not the mean. Cash games are won by the
             lineup that clears the line, so play the mean. Tournaments pay the
             tail, so play the ceiling. A player with mean 11 / ceiling 26 and
             a player with mean 13 / ceiling 17 are different bets, and the
             mean column alone cannot tell them apart.

  swap_pts   what you GIVE UP in projected points by benching the lineup's
             marginal player at that position for this one. Almost always
             negative -- the optimizer already took the best legal option, so
             every override has a price. The question is whether it is small.

  swap_$     salary FREED by the same swap. Positive means the move pays for
             itself and hands you money for another slot.

  cap_after  leftover cap once the swap is done. NEGATIVE MEANS ILLEGAL -- you
             cannot afford him without breaking something else too. This is the
             column that stops a plausible-looking swap from wasting your time.

  status     DK's injury tag, and depth_rank next to it. A large positive value
             on a player listed Q or buried at depth 3 is usually not a deal,
             it is the model not knowing he is hurt. Deals and injuries look
             identical in a value column, which is exactly why the flag sits
             next to it.

WHY THE SWAP IS SCORED AGAINST THE *MARGINAL* PLAYER

For each position the comparison is against the lowest-projected rostered
player at that position -- the one you would actually cut. Comparing against
the best player in the slot would be answering a question nobody asks, since
you are not benching your stud to plug a value play.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SALARY_CAP = 50_000
POSITIONS = ["QB", "RB", "WR", "TE", "DST"]

# FLEX means an RB/WR/TE candidate competes with everyone rostered in his own
# position group, including whoever happens to be occupying the flex slot. The
# optimizer never labels the flex separately, so grouping by position is
# already correct -- a WR simply has 3 or 4 incumbents depending on the lineup.


def build(pool: pd.DataFrame, lineup: pd.DataFrame,
          proj_col: str = "projection") -> dict[str, pd.DataFrame]:
    """One board per position. Returns {position: DataFrame}, richest first."""
    pool = pool.copy()
    rostered = set(lineup["name"])
    leftover = SALARY_CAP - int(lineup["salary"].sum())

    pool["in_lineup"] = pool["name"].isin(rostered)
    if "ceiling" not in pool.columns:
        pool["ceiling"] = np.nan
    if "value" not in pool.columns and "salary_implied_pts" in pool.columns:
        pool["value"] = pool[proj_col] - pool["salary_implied_pts"]

    boards = {}
    for pos in POSITIONS:
        cand = pool[pool["position"].eq(pos)].dropna(subset=[proj_col]).copy()
        if cand.empty:
            continue

        inc = lineup[lineup["position"].eq(pos)]
        if inc.empty:
            # no one rostered here (shouldn't happen in a legal lineup, but a
            # partial/manual lineup is a fair thing to hand this function)
            cand["vs"] = None
            cand["swap_pts"] = np.nan
            cand["swap_$"] = np.nan
            cand["cap_after"] = leftover
        else:
            marginal = inc.loc[inc[proj_col].idxmin()]
            cand["vs"] = marginal["name"]
            cand["swap_pts"] = cand[proj_col] - marginal[proj_col]
            cand["swap_$"] = int(marginal["salary"]) - cand["salary"]
            cand["cap_after"] = leftover + cand["swap_$"]
            # Blank the swap columns for EVERY rostered player, not just the
            # marginal one. "Swap Gibbs for Jeanty" is not a move you can make
            # -- they are both already in the lineup -- and printing a number
            # there invites reading it as an upgrade.
            already = cand["in_lineup"]
            cand.loc[already, ["vs", "swap_pts", "swap_$"]] = [None, np.nan, np.nan]
            cand.loc[already, "cap_after"] = leftover

        # DST never gets `salary_implied_pts` (it comes through the Vegas
        # heuristic, not the model), so its value column is all-NaN and sorting
        # on it would order the board arbitrarily. Fall back to raw projection.
        by = "value" if cand["value"].notna().any() else proj_col
        boards[pos] = cand.sort_values(by, ascending=False)

    return boards


COLS = ["name", "team", "salary", "projection", "ceiling", "value",
        "status", "depth_rank", "vs", "swap_pts", "swap_$", "cap_after"]


def show(boards: dict[str, pd.DataFrame], top: int = 12,
         proj_col: str = "projection") -> None:
    """Print each board: the best deals, plus every rostered player regardless.

    Rostered players are always shown even when their value rank is poor --
    otherwise the board tells you what to plug IN without ever showing you what
    you would be plugging OUT.
    """
    for pos, b in boards.items():
        keep = b.head(top)
        # always surface the lineup's own players, even if they rank below `top`
        missing = b[b["in_lineup"] & ~b.index.isin(keep.index)]
        # reindex through `b` rather than re-sorting: build() already picked the
        # right sort key for this position, and DST's is not the same as WR's.
        shown = set(keep.index) | set(missing.index)
        view = b.loc[[i for i in b.index if i in shown]]

        cols = [c for c in COLS if c in view.columns]
        if proj_col != "projection" and proj_col in view.columns:
            cols = [proj_col if c == "projection" else c for c in cols]

        disp = view[cols].copy()
        disp.insert(0, "", np.where(view["in_lineup"], ">>", ""))
        for c in ["projection", proj_col, "ceiling", "value", "swap_pts"]:
            if c in disp.columns:
                disp[c] = disp[c].astype(float).round(1)
        for c in ["salary", "swap_$", "cap_after"]:
            if c in disp.columns:
                disp[c] = disp[c].astype("Int64")

        n_in = int(view["in_lineup"].sum())
        print(f"\n--- {pos} ---  ({n_in} rostered, showing {len(disp)} of {len(b)})")
        print(disp.to_string(index=False, na_rep=""))

    print("\n  >> = in the optimizer's lineup")
    print("  swap_pts / swap_$ are versus the 'vs' player (the one you'd cut)")
    print("  cap_after < 0 means you cannot afford the swap on its own")


def to_excel(boards: dict[str, pd.DataFrame], lineup: pd.DataFrame,
             path: Path, proj_col: str = "projection") -> Path:
    """One sheet per position plus the lineup, for reading away from a terminal."""
    from openpyxl.utils import get_column_letter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = pd.ExcelWriter(path, engine="openpyxl")
    except PermissionError:
        # open in Excel right now -- write beside it rather than dying
        path = path.with_name(f"{path.stem}_new{path.suffix}")
        writer = pd.ExcelWriter(path, engine="openpyxl")

    with writer:
        lu_cols = [c for c in ["name", "position", "team", "game", "salary",
                               proj_col, "ceiling", "status"] if c in lineup.columns]
        lineup[lu_cols].to_excel(writer, sheet_name="lineup", index=False)

        for pos, b in boards.items():
            cols = [c for c in COLS + ["in_lineup"] if c in b.columns]
            b[cols].to_excel(writer, sheet_name=pos, index=False)

        for ws in writer.sheets.values():
            ws.freeze_panes = "A2"
            for col in ws.columns:
                vals = [len(str(c.value)) for c in col if c.value is not None]
                ws.column_dimensions[get_column_letter(col[0].column)].width = \
                    min(max(vals or [8]) + 3, 30)

    return path

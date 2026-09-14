# NFL DFS — Weekly Points Model + Lineup Optimizer

Predict DraftKings fantasy points for every player on the Sunday main slate, then
build the optimal lineup under the $50,000 salary cap.

Built as an AI/ML engineering portfolio project alongside PMLE cert prep. The
model is deliberately simple; the **pipeline around it is deliberately overbuilt**,
because the lifecycle is the skill being practiced.

## Architecture

Three layers, kept separate on purpose:

1. **Projection model (ML).** Predicts points per player. Gradient-boosted trees.
2. **Optimizer (deterministic).** Integer program: maximize projected points
   subject to cap, roster slots, and stacking rules. No ML here — it should be
   exactly reproducible.
3. **Agent (later).** Reconciles late-breaking gameday information — inactives,
   weather, conflicting beat reports — into projection adjustments before the
   optimizer runs.

Being able to explain *why layer 2 is not an agent* is half the point of the
project.

## Data sources — all free, no API keys

| Source | What it gives | Notes |
|---|---|---|
| nflverse (`nfl_data_py`) | weekly box scores, schedules, snaps, injuries, depth charts | The labels. 49k player-weeks, 2016-2024. |
| nflverse schedules | Vegas spread + total, weather, roof, rest | **100% coverage 2016-2025** |
| DraftKings draftables | salaries, positions, injury status, FPPG, opponent rank | Unofficial endpoint — see caveat below |

**DraftKings caveat.** There is no official DK API. The endpoints used here are
unauthenticated but undocumented and unsupported; they can change shape without
notice. All DK-specific parsing is isolated in `src/dk_client.py` so that when
they do, exactly one file breaks. Personal research use only — pulled data is
gitignored and is not redistributed.

## Roster rules (DK NFL Classic)

9 players, $50,000 cap: QB, RB, RB, WR, WR, WR, TE, FLEX (RB/WR/TE), DST.

DK lists each RB/WR/TE twice — once in its own slot and once as FLEX. `dk_client`
dedupes on `player_dk_id`; skipping that step double-counts every flex-eligible
player.

## Layout

```
nfl-dfs/
  FEATURES.md          <- the feature spec. Start here.
  src/
    dk_client.py       <- DK slate ingestion (isolated). Timestamped snapshots.
    scoring.py         <- DK Classic scoring computed from raw stats
    fetch_nflverse.py  <- historical pull
    features.py        <- feature engineering       (next)
    train.py           <- model + time-ordered backtest (next)
    optimize.py        <- salary-cap integer program  (next)
  data/
    raw/               <- nflverse pulls
    slates/            <- timestamped DK snapshots, one dir per draft group
```

## Environment

Uses the existing `ffml` conda env (Python 3.11 — 3.13 lacks wheels for some deps).

```
conda activate ffml
pip install pulp                     # optimizer; not yet installed
python src/fetch_nflverse.py         # one-time history pull
python src/dk_client.py --list       # see available slates
python src/dk_client.py              # snapshot the Sunday main slate

# full weekly run
python src/features.py                                    # rebuild model table
python src/train.py                                       # walk-forward validate
python src/predict_slate.py --slate <players_*.parquet>        --season 2026 --week 1 --source dk_vegas
python src/optimize.py --slate <projections_*.parquet> --n 3
```

## Week 1 caveat

Every rolling feature is empty before any games are played, so Week 1 uses
`--source dk_vegas`: DK's own FPPG scaled by the Vegas implied team total, with a
15% haircut on Questionable tags. The trained model takes over around Week 4 once
in-season form exists. The projection source is stamped on every row so a Week 1
lineup is never mistaken for a model output.

## Status

- [x] DK ingestion working — live-tested, Week 1 2026 slate: 744 players, 12 games
- [x] Historical pull — 49,161 player-weeks, 2,761 games, 253k snap rows
- [x] DK scoring validated (+2.14 pts on avg top-decile week vs. generic PPR)
- [x] Feature spec written — `FEATURES.md`
- [x] Feature engineering with leakage guards — 45,715 rows x 43 features
- [x] Walk-forward backtest — slate-rank Spearman 0.596 vs 0.579 rolling-avg baseline
- [x] Heuristic backtests — see FEATURES.md section 9 (TE buy-low is the live finding)
- [x] Lineup optimizer — ILP w/ stacking, team caps, DST anti-correlation
- [x] End-to-end: live slate -> projections -> lineups
- [ ] Rebuild 2025 weekly stats from play-by-play (nflverse weekly stops at 2024)
- [x] Gameday report -> Discord (`src/gameday_report.py`) — availability, weather, news
- [x] Slate watcher (`src/watch_slate.py`) — diffs status changes, filters to what matters
- [x] Lineup tracking (`src/track.py`) — scores lineups from public box scores, no DK account
- [x] Containerized + Cloud Build config — see `deploy/DEPLOY.md`
- [ ] Deploy: Cloud Run job + Cloud Scheduler, Sun 12:00 ET (commands ready, not yet run)
- [ ] Sequence-model (DL) benchmark vs the GBM
- [ ] Port to a KFP pipeline w/ custom training container
- [ ] Weekly scheduled run + drift monitoring

## Why the pipeline is overbuilt

The model trains in seconds and does not need managed infrastructure. It runs on
one anyway — custom container, pipeline components, model registry, scheduled
retraining, drift monitoring — because that lifecycle is the thing worth
practicing, and a project that runs every week for a whole season produces a real
drift story rather than a hypothetical one.

**Naming note:** Google renamed parts of the Vertex AI surface in April 2026.
Build against the current console and current official docs, and record the names
you actually see rather than the ones in older tutorials.

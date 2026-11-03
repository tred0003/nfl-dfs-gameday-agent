# Feature Set — NFL DFS Weekly Points Model

Every term gets a plain-English definition before it gets used. Nothing here
assumes prior ML vocabulary.

---

## 0. What the model actually predicts

**Label** (the thing we are trying to guess): how many DraftKings points a given
player scores in a given game.

One row = one player, one week. `src/scoring.py` computes it from raw box-score
stats rather than using the generic `fantasy_points_ppr` column from nflverse,
because DK pays 3-point bonuses for 300 pass / 100 rush / 100 receiving yards
that the generic column omits.

Measured on 49,161 player-weeks (2016-2024): the bonus is worth **+2.14 points
on the average top-decile week, and 82.9% of those weeks earn one**. Train on
the generic column and you systematically under-rank the ceiling plays — which
are the only ones that matter in a tournament.

**A note on what to predict.** Predicting the *mean* (expected points) is right
for cash games, where you want a safe floor. Tournaments pay the tail, so what
you actually want there is the *ceiling* — the 80th/90th percentile outcome.
Start with the mean because it is simpler; add quantile prediction later. This
distinction matters more than it looks.

---

## 1. Opportunity — the single most predictive family

Plain English: fantasy points come from touching the ball. A mediocre player who
gets 12 targets beats an efficient one who gets 3. Volume is stable week to week;
efficiency is not. **If you only build one family, build this one.**

| Feature | Source | Meaning |
|---|---|---|
| `targets`, `carries` | weekly | Raw opportunity count |
| `target_share` | weekly | Share of team targets — already normalized for pace |
| `air_yards_share` | weekly | Share of team intended passing yards; captures *downfield* role |
| `wopr` | weekly | Weighted combo of target share + air yards share; standard opportunity index |
| `offense_pct` | snaps | % of team snaps played — the "is he actually on the field" check |
| `racr`, `pacr` | weekly | Efficiency per air yard. Use as adjustment, never as primary |

**Derived, and worth the effort:**

- `targets_per_snap` — separates a full-time decoy from a part-time weapon
- Red-zone opportunity (touches inside the 20) — needs play-by-play, adds
  meaningful TD signal. Phase 2.

---

## 2. Game environment (Vegas) — free, complete, and nearly as strong

Plain English: the betting market is a very good forecast of how many points a
team will score. If Vegas expects Detroit to score 30, Detroit skill players are
in a good spot. This is the highest value-per-effort family in the project.

**Coverage is 100% for 2016-2025** in `schedules.parquet`. No scraping needed.

The key derived feature:

```
implied_team_total = total_line / 2 + spread_line / 2    # home team
implied_team_total = total_line / 2 - spread_line / 2    # away team
```

(Verified: 2024 Wk1 BAL@KC, spread 3.0, total 46.0 → KC 24.5, BAL 21.5.)

| Feature | Meaning |
|---|---|
| `implied_team_total` | Expected points for THIS player's team. The workhorse. |
| `total_line` | Expected combined score — proxy for overall game pace |
| `spread_line` | Margin. Drives **game script**: big favorites run the ball late (RB up, WR down); big underdogs throw constantly (WR/TE up, RB down) |
| `implied_opp_total` | How much the opponent is expected to score — high means shootout |
| `is_home` | Small but real |

Game script is where a lot of amateur models stop and where yours should not:
interact spread with position. A -10 favorite's RB and a +10 underdog's WR are
both good plays, for opposite reasons.

---

## 3. Player form / baseline

Plain English: what has this player been doing lately.

- Rolling mean of DK points over last 3, 5, and 8 games
- Rolling mean of the *opportunity* metrics above (more stable than points)
- Exponentially weighted mean (recent games count more)
- Games played this season — a proxy for "do we have enough data to trust this"

**Every one of these must use only games strictly BEFORE the week being
predicted.** See section 8.

---

## 4. Matchup

Plain English: some defenses are much easier to score on, and they are often
easier against one position specifically.

- `dk_oprk` — DK ships opponent rank vs. position right in the salary feed. Free.
- Rolling DK points allowed by the opposing defense to that position over the
  prior N weeks — build this yourself; it is more granular than OPRK.
- Caveat worth knowing: defense-vs-position is noisy early in a season and is
  partly a reflection of *which offenses that defense happened to face*. Weight
  it lower than opportunity and Vegas.

---

## 5. Availability and vacated opportunity — where your edge probably lives

Plain English: when a starter is ruled out, his targets do not vanish. They go to
whoever is next on the depth chart. The market is slow to reprice that, because
salaries were set days earlier.

This is the family most likely to encode the instinct you have been winning with.

| Feature | Source |
|---|---|
| `status` (None / Q / OUT / IR) | DK draftables — live |
| `news_status` (None / Recent / Breaking) | DK draftables — flags late-breaking news |
| Practice participation (DNP / Limited / Full) | injuries.parquet |
| **Teammate targets vacated** | derived: sum of prior target share of teammates now OUT |
| **Depth chart position** | depth_charts.parquet — who moves up |

`salary` was set before the injury news. A player whose target share is about to
jump but whose price has not moved is *precisely* the inefficiency DFS pays for.

---

## 6. Weather

Plain English: wind wrecks the passing game; almost nothing else matters much.

- `roof` (outdoors / dome / closed) and `is_dome` from the DK feed
- `wind` — the one that matters. Above ~15 mph, downgrade passing and deep WRs.
- `temp` — weaker than people assume; include, do not lean on it
- `surface`

Coverage note: `temp` and `wind` are 65% populated across 2016-2025 — the missing
35% is essentially all domes. Fill those as "no wind, controlled temp" rather than
treating them as missing data, or you will teach the model something false.

---

## 7. Price and value

Salary is not a feature that predicts points — it is a *constraint*. Keep it out
of the model that predicts scoring, then use it in the optimizer.

The exception: **salary movement week over week** is legitimate signal, because it
reflects DK's own reaction to news. That requires your own snapshot history, which
is why `dk_client.py` timestamps every pull starting now.

---

## 8. Leakage rules — read before writing any feature code

"Leakage" = accidentally letting the model see information that would not have
existed at decision time. It produces backtests that look brilliant and lineups
that lose money. Three rules:

1. **Shift everything.** Rolling features are computed per player, then shifted
   one game forward. A Week 8 row may only contain data through Week 7.
2. **Split by time, never randomly.** Train on earlier weeks, test on later ones.
   A random shuffle lets the model see the future.
3. **Timestamp your inputs.** Injury status changes hourly on Sunday morning. A
   model trained on final status but deployed Saturday night is being graded on
   information it will not have. This is why slate pulls are snapshotted.

---

## 9. Your heuristics — backtested 2016-2024

Five rules, tested against 47,071 regular-season player-weeks in
`research/test_heuristics.py`. **Read 9.6 before concluding anything** — every
test below measures POINTS, and DFS pays POINTS PER DOLLAR.

### 9.1 WR bounce-back — not supported as a points effect

*"A high-volume WR1/WR2 with 1-2 bad games is due; the QB force-feeds him."*

Outcome = next-game DK points minus that player's own prior 4-game mean:

| Prior 4-wk target share | After a bad game | Control | Difference |
|---|---|---|---|
| >= 0.18 | -1.41 (n=1139) | -0.95 (n=3474) | **-0.45** |
| >= 0.22 | -1.83 (n=653) | -1.52 (n=2162) | **-0.31** |
| >= 0.25 | -2.57 (n=362) | -2.13 (n=1415) | **-0.44** |
| >= 0.28 | -3.06 (n=169) | -2.47 (n=765) | **-0.59** |

Negative at every threshold — a bad game predicts a slightly *worse* next game,
not a better one. Two consecutive bad games doesn't rescue it. Splitting on
whether volume held up while points dropped (bad luck) versus volume dropping too
(role change) also produced no bounce-back: -1.54 vs -1.34.

The direction is consistent across every cut, which is what makes it credible;
any single row's magnitude is within noise.

### 9.2 Rookie/backup QB force-feeds the WR1 — supported, but **inverted**

*Green QB* = 5 or fewer career starts, 2018-2024.

| | WR1 target share | WR1 targets | WR1 DK points |
|---|---|---|---|
| Veteran QB | 0.236 | 7.68 | **14.69** (n=2662) |
| Green QB | 0.228 | 6.98 | **11.38** (n=349) |

WR1s score **22% less** with an inexperienced QB. Target *share* is essentially
flat, so the concentration you describe is roughly neutral — but the pie is
smaller and less efficient, and that dominates. Largest effect in the test set,
and it points the opposite way from the rule: **fade WR1s with green QBs.** An
inverted rule is still an edge.

### 9.3 Green QB teams run more — technically true, not actionable

Run rate 0.451 vs 0.438. But carries are 26.83 vs 26.67 — nearly identical. The
run-rate gap comes from *fewer pass attempts*, not more carries. Nothing for an RB
to gain. Good instinct to have left this one alone.

### 9.4 Bad weather boosts RBs — mechanism real, payoff absent

No precipitation column exists in nflverse schedules, so this tests **wind**, not
rain. Outdoor games only:

| Wind | Run rate | Carries | Pass attempts |
|---|---|---|---|
| 0-5 mph | 0.431 | 26.26 | 34.69 |
| 6-10 | 0.440 | 26.80 | 34.04 |
| 11-15 | 0.444 | 26.86 | 33.70 |
| 15+ | 0.449 | 27.08 | 33.14 |

Cleanly monotonic — wind really does shift play-calling toward the run. But RB
production doesn't follow: **9.07 DK points in wind >15mph vs 9.19 otherwise.**
The extra 0.8 carries are cancelled by a worse offensive environment overall.

Rain is untested and genuinely different from wind. Testing it needs an external
historical weather source (Meteostat or NOAA) joined on stadium coordinates. Worth
doing — this is the one rule the data hasn't had a fair chance to refute.

### 9.5 Fade elite WR/TE vs. a top secondary — partially testable

The team-level version (rolling DK points allowed to the position) is buildable
from data already on disk — see section 4. The **shadow-corner** version, where a
specific CB travels with a specific WR, needs matchup data that isn't free.
Implement the team version; leave the corner version as a manual override.

### 9.6 The reframe that matters

Every test above measures **points**. DFS pays **points per dollar**, and DK cuts
a player's salary after a bad game.

At the >=0.22 threshold, post-bad-game WR1s scored 14.36 vs 15.32 for the control
— a **6.3% points deficit**. So the rule wins if and only if DK drops the price by
more than ~6.3%. On a $7,000 WR that is a $440 cut, well inside the normal weekly
range for a dud performance.

**The instinct may be entirely correct and completely invisible to these tests.**
Confirming it requires historical salary data (see README — `nfldfs` or RotoGuru),
which is now a priority rather than a nice-to-have. Until that lands this is an
open question, not a dead rule.

That is the general lesson of the project: measure the quantity the contest
actually pays.

### 9.7 Buy low, tested properly — the rule is right, the position is wrong

You confirmed the two league wins came from salary-cap DFS and the edge was
"buy low when the value comes down." So the fair test is value, not points.

Backfilled DK salaries 2016-2021 from RotoGuru (`src/fetch_salaries.py`, 41,201
priced player-weeks; the archive stops after 2021). It carries salary and actual
points in the same row, so this needs no join.

"Bad game" = scored under 60% of the player's own prior 4-week average.

| Position | Prior salary | Price cut | Points change | Value |
|---|---|---|---|---|
| WR | >= $6,000 | -4.3% | **-8.7%** | -3.5% |
| WR | >= $6,500 | -3.9% | **-6.7%** | -2.6% |
| WR | >= $7,000 | -3.5% | **-6.9%** | -3.5% |
| RB | >= $6,000 | -5.6% | **-15.5%** | -8.2% |
| RB | >= $7,000 | -5.2% | **-15.2%** | -10.2% |
| TE | >= $5,000 | -6.3% | **+11.1%** | **+22.9%** |

**WR and RB: the price does not fall far enough.** DK cuts about 4-6%, but
production falls 7-15%. The mechanism you identified is real — the price *does*
come down — it just does not come down enough to pay for the drop in output. RBs
are the worst version: a 5% discount on a 15% production collapse.

Formally these gaps are not distinguishable from zero (WR gap -0.42 pts vs the
price baseline, p=0.51; RB -0.69, p=0.30). The honest claim is **no evidence of
an edge, with point estimates leaning negative** — not that it actively loses.

**TE is the exception, and it is a real one.** Points *rise* 11% after a dud while
DK cuts the price 6.3%. Measured against what that price normally delivers:

| TE prior salary | After bad game | Control | Gap | p |
|---|---|---|---|---|
| >= $4,000 | +0.80 (n=229) | -0.14 (n=585) | +0.94 | 0.138 |
| >= $4,500 | +0.92 (n=165) | -0.28 (n=415) | +1.20 | 0.138 |
| >= $5,000 | +2.00 (n=109) | -0.34 (n=277) | +2.34 | **0.031** |
| >= $5,500 | +1.57 (n=74) | -0.39 (n=199) | +1.96 | 0.155 |
| >= $6,000 | +4.15 (n=42) | -0.25 (n=142) | +4.40 | **0.027** |

Positive at every threshold and growing with price. Caveat, stated plainly: the
thresholds are nested subsets, so these are not five independent tests, and the
cells are small. **Suggestive and worth a real allocation; not established.**

Likely mechanism: TE scoring is far more touchdown-dependent than WR or RB, so a
dud week is more often variance than a role change — while DK prices TEs off
recent points, cutting hard on the same noise.

### 9.8 Severity matters more than the rule itself

WRs priced >= $6,500, by how badly they missed their own form:

| Last week vs. own form | Price change | Next-week points | vs. price baseline |
|---|---|---|---|
| < 30% of form | -5.1% | 14.48 | **-1.25** |
| 30-50% | -3.8% | 16.18 | -0.13 |
| 50-70% | -2.8% | 16.93 | +0.43 |
| 70-100% | -1.4% | 17.61 | **+0.74** |
| beat form | +1.7% | 16.84 | -0.32 |

Both extremes lose. Severe duds are usually a real problem — injury, benching,
role loss — and DK does not discount enough to compensate. Players coming off a
big game get marked up too far. **The sweet spot is the mild dip: a player at
70-100% of his form whose price ticked down slightly.**

That is a sharper rule than the one you started with, and it came out of your
instinct rather than replacing it.

### 9.9 Net rewrite of your rules

1. **Buy low on TEs**, especially expensive ones, after a bad game. Best-supported
   finding in the project.
2. **Do not buy low on RBs.** Strongest negative. A dud RB is usually a hurt or
   benched RB.
3. **WR buy-low: neutral.** Use mild-dip severity, not the blowups.
4. **Fade WR1s with green QBs** (9.2) — your rule, inverted.
5. **Wind shifts play-calling but not RB points** (9.4). Rain still untested.

---

## 10. Deliberately out of scope for v1

Not oversights — parked, with reasons:

- **Ownership projection.** What % of entries roster a player. Essential for
  tournament strategy, but there is no free historical source.
- **Play-by-play derived features** (red zone, route participation). High value,
  heavy pull. Phase 2.
- **Beat-reporter NLP.** The text version of your body-language instinct. Belongs
  in the agent layer, later.
- **Correlation / stacking.** A QB and his WR score together, so lineup
  construction should treat them jointly. This is an *optimizer* concern, not a
  feature — the model predicts players independently; the optimizer stacks them.

"""Salary lag: after a player beats his salary N weeks in a row, does DK reprice fast enough?

The hypothesis (Adam Thielen, 2023): DK is slow to raise a cheap player who keeps
producing, so a streak of beating salary tends to continue for a while.

"Expected points" at a salary = a straight line of points on salary, fit per
position-season. Fit on the same season, so this is DESCRIPTIVE research, not a
feature -- a model version must use only earlier weeks.

Run: python research/test_value_streaks.py
"""
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src'))
import numpy as np, pandas as pd
from compare_batch import fantasydata_benchmark

# 2016-2021: RotoGuru (salary + actual points in one row)
rg = pd.read_parquet('data/raw/dk_salaries.parquet')
rg = rg[rg.position.isin(['QB', 'RB', 'WR', 'TE'])].dropna(subset=['salary'])
rg['key'] = rg.name + '|' + rg.team
rg = rg[['key', 'season', 'week', 'position', 'salary', 'dk_points']]
# 2022-2024: FantasyData salary + actual points from the stats table
fd = fantasydata_benchmark()
m = pd.read_parquet('data/processed/model_table.parquet')[['player_id', 'season', 'week', 'position', 'dk_points']]
fd = fd.merge(m, on=['player_id', 'season', 'week'])
fd['key'] = fd.player_id
fd = fd[['key', 'season', 'week', 'position', 'salary', 'dk_points']]
d = pd.concat([rg, fd], ignore_index=True)
d['position'] = d.position.astype(str)
d = d[d.salary >= 3000]  # skip minimum-price backups

# expected points at a salary: straight line per position-season (descriptive only)
def fit(g):
    b = np.polyfit(g.salary, g.dk_points, 1)
    return pd.Series(np.polyval(b, g.salary), index=g.index)
d['expected'] = d.groupby(['position', 'season'], group_keys=False).apply(fit)
d['value'] = d.dk_points - d.expected
d = d.sort_values(['key', 'season', 'week'])
g = d.groupby(['key', 'season'], sort=False)

BEAT = 6  # beat salary-implied points by 6+
d['beat'] = (d.value >= BEAT).astype(int)
# consecutive beats in the games BEFORE this one
prev = g.beat.shift(1).fillna(0)
d['streak'] = prev.groupby((prev == 0).cumsum()).cumsum()
d['streak'] = np.where(g.cumcount() == 0, 0, d['streak'])
d['salary_change_pct'] = (d.salary / g.salary.shift(1) - 1) * 100
d['streak_b'] = pd.cut(d.streak, [-1, 0, 1, 2, 3, 99], labels=['0 (none)', '1 week', '2 in a row', '3 in a row', '4+ in a row'])

print(f'{len(d):,} player-weeks, 2016-2024, salary >= $3,000. "Beat" = {BEAT}+ pts above salary-implied.\n')
t = d.dropna(subset=['salary_change_pct']).groupby('streak_b').agg(
    player_weeks=('value', 'size'),
    salary_change_this_week_pct=('salary_change_pct', 'mean'),
    avg_pts_above_salary=('value', 'mean'),
    beat_again_pct=('beat', lambda x: x.mean() * 100))
print(t.round(2).to_string())
base = d.beat.mean() * 100
print(f'\nbase rate of beating salary by {BEAT}+ in any week: {base:.1f}%')

print('\nby position, after 2+ in a row:')
x = d[d.streak >= 2].groupby('position').agg(n=('value', 'size'), pts_above=('value', 'mean'),
                                              beat_again=('beat', lambda s: s.mean() * 100),
                                              salary_chg=('salary_change_pct', 'mean'))
x['base_beat'] = d.groupby('position').beat.mean() * 100
print(x.round(1).to_string())

print('\nby era (after 2+ in a row): is DK getting faster?')
d['era'] = np.where(d.season <= 2019, '2016-19', np.where(d.season <= 2021, '2020-21', '2022-24'))
y = d[d.streak >= 2].groupby('era').agg(n=('value', 'size'), pts_above=('value', 'mean'),
                                         beat_again=('beat', lambda s: s.mean() * 100),
                                         salary_chg=('salary_change_pct', 'mean'))
print(y.round(1).to_string())

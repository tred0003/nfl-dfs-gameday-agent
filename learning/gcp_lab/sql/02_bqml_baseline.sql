-- Stage 2b: the dumb baseline, trained inside BigQuery with SQL only.
-- Linear regression on 4 obvious columns first; boosted tree after.
CREATE OR REPLACE MODEL `fantasy-football-ml-503718.nfl_lab.bqml_linear`
OPTIONS (model_type = 'linear_reg', input_label_cols = ['dk_points']) AS
SELECT dk_points, dkp_r5, salary, implied_team_total, position
FROM `fantasy-football-ml-503718.nfl_lab.v_split` WHERE split = 'TRAIN';

CREATE OR REPLACE MODEL `fantasy-football-ml-503718.nfl_lab.bqml_bt`
OPTIONS (model_type = 'boosted_tree_regressor', input_label_cols = ['dk_points'],
         max_iterations = 200, learn_rate = 0.05) AS
SELECT * EXCEPT (player_id, player_display_name, recent_team, opponent_team, split, season, week)
FROM `fantasy-football-ml-503718.nfl_lab.v_split` WHERE split = 'TRAIN';

-- Score on 2024 (VALID). ML.EVALUATE returns RMSE/MAE/R2; rank correlation is done in 06_compare.py.
SELECT * FROM ML.EVALUATE(MODEL `fantasy-football-ml-503718.nfl_lab.bqml_bt`,
  (SELECT * EXCEPT (player_id, player_display_name, recent_team, opponent_team, split, season, week)
   FROM `fantasy-football-ml-503718.nfl_lab.v_split` WHERE split = 'VALID'));

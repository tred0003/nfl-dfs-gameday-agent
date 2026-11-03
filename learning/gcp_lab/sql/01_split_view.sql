-- Stage 2a: one view that stamps every row with its split. Time-based, never random.
CREATE OR REPLACE VIEW `fantasy-football-ml-503718.nfl_lab.v_split` AS
SELECT *,
  CASE WHEN season <= 2023 THEN 'TRAIN'
       WHEN season = 2024  THEN 'VALID'
       WHEN season = 2025  THEN 'FINAL' END AS split
FROM `fantasy-football-ml-503718.nfl_lab.model_table_v3`
WHERE dk_points IS NOT NULL AND season <= 2025;

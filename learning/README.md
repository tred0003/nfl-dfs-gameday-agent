# learning/

Side experiments run to learn a specific Google Cloud or TensorFlow tool on this
project's real data. They are not part of the production system.

| Folder | What it is | Status |
|---|---|---|
| `gcp_lab/` | The same points-prediction problem run through BigQuery ML, Vertex AI Workbench, Vertex AI Experiments and AutoML, to compare them against the production XGBoost model | Scaffolding; most stages not run |
| `tf_sequence/` | A TensorFlow sequence model over each player's recent games, benchmarked against XGBoost, with a `tf.data` pipeline study | Built and benchmarked; did not beat XGBoost |

The production model, agent and web app live in `src/`, `gameday_agent/` and `webapp/`.

# Example request payloads

## `instances-minimal.json`

**This payload is deliberately wrong, and it is kept because it is wrong.**

Three features out of the eighty-three the model card pins. `align()` fills the
other eighty with NaN, XGBoost reads NaN as a real value, and the server answers
with a confident number near the pool mean and a 200. Nothing is red.

That was invisible until serving started reporting coverage. It now answers:

```json
{"mean": 9.55, "ceiling": 13.07, "mean_raw": 8.94, "calibrated": true,
 "features_present": 6, "features_expected": 83}
```

`features_present: 6` — the three sent, plus the four `pos_` dummies rebuilt
from `position`, minus `position` itself which is not a feature. Six of
eighty-three is the whole story, and it is the reason the field exists.

Use it to smoke-test that a container is up and answering. Do not use it to
judge whether the model is any good.

## A real payload

One line of a batch input file:

```
python src/export_batch_input.py --season 2025 --week 17
head -1 data/batch/2025_wk17_input.jsonl
```

Those score at `features_present` 70–83.

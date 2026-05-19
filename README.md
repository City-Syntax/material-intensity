# Material Intensity Predictor Web App

A Streamlit app that queries the **FinalQueryModel** two-part pipeline to estimate
building material intensities (kg/m²) from archetype features.

Web App: https://predictmi.streamlit.app/.

## App Layout

**Per-material column cards** (one per material: Concrete, Glass, Steel, Wood, Brick):
- `5th / Median / 95th percentile (kg/m²)` — conditional intensity intervals
- `Database availability` — `p_recorded` as a percentage

**Building-level data support** (single expander, below the columns):
- Overall `Data support` score (minimum across materials) with a ⚠️ warning if any material is sparsely supported
- Per-material breakdown table: `n_support` and `support_score`

**Five-panel bar chart** — visual summary of 5th / 50th / 95th percentile per material.

## Query Output Keys

`model.query(X_proc)` returns a dict keyed by material. Each entry contains:

| Key | Type | Description |
|-----|------|-------------|
| `p_recorded` | float (array) | P(MI recorded in database \| features) |
| `probability_unrecorded` | float (array) | `1 − p_recorded` |
| `conditional_p05` | float (array) | 5th-percentile conditional intensity (kg/m²) |
| `conditional_p50` | float (array) | Median conditional intensity (kg/m²) |
| `conditional_p95` | float (array) | 95th-percentile conditional intensity (kg/m²) |
| `recording_adjusted_median` | float (array) | `p_recorded × conditional_p50` |
| `n_support` | int | Training rows with this material recorded |
| `support_score` | float | `min(1, n_support / LOW_OBS_THRESHOLD)` — training-data coverage score |
| `support_warning` | bool | `True` if `n_support < LOW_OBS_THRESHOLD` |
| `query_confidence` | float (array) | `p_recorded × support_score` — combined signal; does **not** alter MI intervals |

`p_recorded` and `support_score` measure different things: one is a per-query
database-recording probability, the other is a per-material training-data coverage
score.

## Input Fields

| Field | Type |
|-------|------|
| Construction period | Numeric (year) |
| Typology | Categorical |
| Primary Code | Categorical |
| Hybrid Structure | Categorical (0 = single, 1 = mixed) |
| Country | Categorical |

## Artifacts

| File | Purpose |
|------|---------|
| `model.joblib` | Trained `FinalQueryModel` (Stage 1 + Stage 2) |
| `preprocessor.joblib` | `ColumnTransformer` — must be applied before `model.query()` |
| `two_stage_model.py` | Class definitions required by `joblib.load` |
| `model_info.json` | Column names, thresholds, split sizes, `n_support` per material |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run Material_Intensity_Predictor.py
```

## Inference (Python)

```python
import joblib
model = joblib.load("model.joblib")
pre   = joblib.load("preprocessor.joblib")
X_proc = pre.transform(X_raw[X_cols])
result = model.query(X_proc)
# result[material] keys:
#   p_recorded, probability_unrecorded,
#   conditional_p05, conditional_p50, conditional_p95,
#   recording_adjusted_median,
#   n_support, support_score, support_warning, query_confidence
```

## Updating Artifacts

After retraining in `prediction_model.ipynb`, copy the new artifacts into this folder:

```
model.joblib
preprocessor.joblib
model_info.json
best_observation_params.json
best_intensity_params.json
evaluation_summary.csv
```

## Troubleshooting

- Port conflict: `streamlit run Material_Intensity_Predictor.py --server.port 8502`
- `joblib.load` error: confirm `two_stage_model.py` is present alongside the `.joblib` files.

# Material Intensity Predictor Web App

This web app serves predictions from the current **FinalQueryModel** pipeline.

Web Link: https://predictmi.streamlit.app/

Model internals:
- Stage 1 (`ObservationModel`): independent `XGBClassifier` per material with Platt calibration (`CalibratedClassifierCV`) → `p_recorded`
- Stage 2 (`IntensityModel`): per-material `XGBRegressor` quantile regression (`reg:quantileerror`) in log1p-space, with inverse-propensity sample weighting from Stage 1 and split-conformal calibration on the validation set → `p05`, `p50`, `p95`

## Quick Start

```bash
pip install -r requirements.txt
streamlit run Material_Intensity_Predictor.py
```

## Model Used

The app loads:
- `preprocessor.joblib`
- `model.joblib`

Model classes are defined in `prediction_model.py` and must be present for `joblib.load` to work correctly.

Predictions include, for each material:
- `p_recorded` — probability the material intensity is recorded in the database
- `support_confidence` — alias of `p_recorded`, shown as data-support/confidence indicator
- `p05` — 5th percentile, conformally calibrated (kg/m²)
- `p50` — median / point estimate (kg/m²)
- `p95` — 95th percentile, conformally calibrated (kg/m²)
- `expected_reported` — `p_recorded × p50` (expected reported intensity, kg/m²)

Prediction intervals (`p05`, `p95`) are produced by Stage 2 quantile regressors and post-hoc expanded by per-material conformal offsets estimated on the held-out validation set to target 90% nominal coverage.

## Input Fields

| Field | Type |
|---|---|
| Construction period | Numeric (year) |
| Typology | Categorical |
| Primary Code | Categorical |
| Hybrid Structure | Categorical |
| Country | Categorical |

## Notes

- The app does not use the legacy PyTorch checkpoint pipeline.
- If artifacts are retrained in the notebook or script, replace `preprocessor.joblib` and `model.joblib` in the project root before restarting the app.

## Troubleshooting

- If port `8501` is occupied: `streamlit run Material_Intensity_Predictor.py --server.port 8502`
- Confirm `preprocessor.joblib`, `model.joblib`, and `prediction_model.py` all exist in the project root.

## Inference API

```python
import joblib
model = joblib.load("model.joblib")
pre   = joblib.load("preprocessor.joblib")
X_proc = pre.transform(X_raw[X_cols])
result = model.query(X_proc)
# result[material] keys: p_recorded, p05, p50, p95, expected_reported,
#                        n_observed_train, coverage_warning
```

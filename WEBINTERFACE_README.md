# Material Intensity Predictor Web App

This web app serves predictions from the current **TwoStageConditionalModel** pipeline.

Model internals:
- Stage 1: `XGBClassifier` per material (`p_presence`)
- Stage 2: `XGBRegressor` per material in log-space (`p50`)
- Joint layer: Gaussian intervals by Typology-specific covariance (`p5`, `p95`)

## Quick Start

```bash
pip install -r requirements.txt
streamlit run Material_Intensity_Predictor.py
```

## Model Used

The app loads:
- `preprocessor.joblib`
- `model.joblib`

Model classes are defined in `two_stage_model.py` and must be present for `joblib.load` to work correctly.

Predictions include, for each material:
- `p5` — 5th percentile (kg/m²)
- `p50` — median / point estimate (kg/m²)
- `p95` — 95th percentile (kg/m²)
- `p_presence` — probability the material appears in the building

The joint uncertainty layer groups buildings by **Typology** to estimate group-specific covariance matrices.

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
- Confirm `preprocessor.joblib`, `model.joblib`, and `two_stage_model.py` all exist in the project root.

# Material Intensity Predictor

This repository packages the current notebook-trained material-intensity model from `prediction_model.ipynb` together with the Streamlit app that serves it.

Web app: https://predictmi.streamlit.app/

## Source Of Truth

The current model is defined, trained, evaluated, and exported from `prediction_model.ipynb`.

The main exported files are:
- `model_finalquery.joblib` - primary serialized `FinalQueryModel`
- `preprocessor.joblib` - fitted preprocessing pipeline
- `model_classes.py` - lightweight class definitions for `joblib` deserialization in the app
- `prediction_model.py` - notebook export of the full training pipeline
- `model_info.json` - saved schema and training metadata
- `evaluation_summary.csv` - held-out test metrics used below

`Material_Intensity_Predictor.py` loads `model_finalquery.joblib` first and falls back to `model.joblib` for backward compatibility.

## Model Overview

The exported model is a two-stage `FinalQueryModel`:

- Stage 1: `ObservationModel` fits one calibrated `XGBClassifier` per material and predicts `P(recorded | x)`.
- Stage 2: `IntensityModel` fits one quantile-regression `XGBRegressor` plus one mean-regression `XGBRegressor` per material in `log1p` space.
- Stage 2 uses inverse-propensity sample weights derived from Stage 1 out-of-fold probabilities.
- The current export keeps `BLEND_MAX_ALPHA = 0.0`, so archetype-mean blending is disabled in the shipped model.

Important interval note:

- `query()` returns the raw Stage 2 quantile outputs `p05`, `p50`, and `p95`.
- Validation-set conformal calibration is computed in the notebook for evaluation reporting only; it is not baked into `model_finalquery.joblib`.

## Input Schema

`model_info.json` records the notebook feature list as:

- `Construction period`
- `Construction period bucket`
- `Typology`
- `Primary Code`
- `Hybrid Structure`
- `Country`
- `Geo_macro`

The saved `preprocessor.joblib` currently transforms these columns:

- numeric: `Construction period`
- categorical: `Construction period bucket`, `Typology`, `Primary Code`, `Country`, `Geo_macro`

Archetype support metadata is computed from:

- `Construction period bucket`
- `Typology`
- `Primary Code`
- `Country`

Implementation note:

- `Hybrid Structure` is present in the notebook feature metadata but is not used by the saved preprocessor.
- The current Streamlit app should supply `Geo_macro` when serving freshly exported notebook artifacts.

## Prediction Outputs

For each material (`Concrete`, `Glass`, `Steel`, `Wood`, `Brick`), `FinalQueryModel.query()` returns:

- `database_reporting_probability` - probability that the material is recorded in the source database
- `p_recorded` - alias of `database_reporting_probability`
- `p05`, `p50`, `p95` - conditional reported-intensity quantiles in kg/m²
- `mean` - conditional reported-intensity mean in kg/m²
- `expected_reported` - `p_recorded * p50`
- `n_observed_train` - number of observed training rows for that material
- `coverage_warning` - `True` when `n_observed_train < 30`
- `archetype_n_train` - number of training rows matching the input archetype
- `archetype_support_level` - one of `none`, `very_low`, `low`, `medium`, `high`

## Current Training Summary

From `model_info.json`:

- `SEED = 42`
- data split: train `1799`, validation `385`, test `386`

Observed training rows per material:

| Material | Observed Train Rows |
|---|---:|
| Concrete | 1385 |
| Glass | 886 |
| Steel | 1638 |
| Wood | 1452 |
| Brick | 1200 |

## Held-Out Test Metrics

From `evaluation_summary.csv`:

| Material | AUC | MAE | RMSE | R2 | Uncal. Cov. | Cal. Cov. |
|---|---:|---:|---:|---:|---:|---:|
| Concrete | 0.963 | 498.65 | 956.64 | 0.153 | 0.835 | 0.882 |
| Glass | 0.966 | 1.15 | 1.79 | 0.400 | 0.803 | 0.870 |
| Steel | 0.984 | 20.15 | 55.30 | 0.222 | 0.824 | 0.905 |
| Wood | 0.953 | 9.90 | 21.19 | 0.517 | 0.863 | 0.895 |
| Brick | 0.943 | 242.05 | 645.71 | 0.166 | 0.884 | 0.924 |

`Cal. Cov.` refers to validation-derived conformal calibration analysed in the notebook, not to interval values returned directly by the exported model.

## Data Sources

The training file `Integrated_MI_database_add_Singapore.xlsx` is integrated from six source streams. Below is the source list in paper-title + DOI format:

1. R: Global construction materials database and stock analysis of residential buildings between 1970-2050  
	DOI: https://doi.org/10.1016/j.jclepro.2019.119146
2. N: Spatiotemporal Characteristics of Global Building Material Intensity Revealed for Circular and Low-Carbon Construction  
	DOI: https://doi.org/10.1021/acs.est.5c05684
3. B: A database seed for a community-driven material intensity research platform  
	DOI: https://doi.org/10.1038/s41597-019-0021-x
4. G: Global Buildings Database Seed on Whole Life Carbon Emissions, Energy Performance, and Material Intensity (GBDB CarbEnMats)  
	DOI: https://doi.org/10.21203/rs.3.rs-3373442/v1
5. C: CBMICD1.0: China's building material intensity coefficient dataset (1949-2015)  
	DOI: https://doi.org/10.1016/j.resconrec.2020.104824
6. Singapore extension records integrated into `Integrated_MI_database_add_Singapore.xlsx`  
	DOI: not available in the current repository metadata

Related publication materials and scripts are in `Nature_Scientific-Data_MI-database/`.

When updating data, keep file naming and column schema consistent with `prediction_model.ipynb` so preprocessing and artifact export remain compatible.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run Material_Intensity_Predictor.py
```

If you regenerate artifacts from `prediction_model.ipynb`, rerun the export cell so these files are refreshed together:

- `model_finalquery.joblib`
- `preprocessor.joblib`
- `prediction_model.py`
- `model_info.json`
- `evaluation_summary.csv`

## Troubleshooting

- If port `8501` is occupied: `streamlit run Material_Intensity_Predictor.py --server.port 8502`
- Confirm `model_finalquery.joblib`, `preprocessor.joblib`, `model_classes.py`, and `model_info.json` exist in the project root.
- If `joblib.load` fails on the model artifact, ensure `model_classes.py` is available so the app can register `ObservationModel`, `IntensityModel`, and `FinalQueryModel` under the `prediction_model` module name.

## Inference Example

```python
import joblib
import pandas as pd

model = joblib.load("model_finalquery.joblib")
pre = joblib.load("preprocessor.joblib")

input_df = pd.DataFrame([
	{
		"Construction period": 2010,
		"Construction period bucket": "post_2010",
		"Typology": "R-SFH",
		"Primary Code": "C",
		"Country": "Singapore",
		"Geo_macro": "Asia",
	}
])

x_proc = pre.transform(input_df)
result = model.query(x_proc, X_raw=input_df)
print(result["Concrete"].keys())
```

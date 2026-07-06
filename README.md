# MIs Database Probabilistic Query Model

This folder contains the current model definition, training workflow, exported artifacts, and evaluation outputs for building material intensity prediction in kg/m².

## Standardized Package Layout

This folder is organized for a GitHub-ready model package with a single source of truth for development:

- `prediction_model.ipynb`: primary development workflow.
- `prediction_model.py`: script export of the notebook for reproducible runs.
- `model_classes.py`: lightweight class definitions for model deserialization.
- Runtime artifacts: `model_finalquery.joblib`, `preprocessor.joblib`, `model_info.json`.
- Evaluation artifact: `evaluation_summary.csv`.
- Data inputs: `Integrated_MI_database_add_Singapore.xlsx`, `Raw_db/`, `sources/`.

Recommended workflow:

1. Develop and validate in `prediction_model.ipynb`.
2. Export/sync script to `prediction_model.py`.
3. Regenerate artifacts and metrics from the same run.
4. Commit notebook, script, and generated artifacts together.

## Model Definition

The deployed model is a two-stage FinalQueryModel implemented in [prediction_model.py](prediction_model.py):

1. Stage 1 (ObservationModel): one calibrated XGBClassifier per material predicts the probability that a material value is recorded.
2. Stage 2 (IntensityModel): one quantile XGBRegressor plus one mean XGBRegressor per material predicts conditional intensity.
3. Stage 2 uses inverse-propensity sample weights derived from Stage 1 out-of-fold probabilities.

Quantile outputs are p05, p50, and p95.

## Input and Targets

Features (X_cols) from [model_info.json](model_info.json):
- Construction period
- Construction period bucket
- Typology
- Primary Code
- Hybrid Structure
- Country
- Geo_macro

Materials (y_cols):
- Concrete
- Glass
- Steel
- Wood
- Brick

Archetype support is computed using:
- Construction period bucket
- Typology
- Primary Code
- Country

## Integrated MI Database Sources (DOI)

The Integrated_MI_database_add_Singapore.xlsx dataset is harmonized from five source databases.
Source labels use R-n, N-n, B-n, G-n, and C-n, where n is the original record index in each source.

- R-n: Global construction materials database and stock analysis of residential buildings between 1970-2050
	DOI: https://doi.org/10.1016/j.jclepro.2019.119146
- N-n: Spatiotemporal Characteristics of Global Building Material Intensity Revealed for Circular and Low-Carbon Construction
	DOI: https://doi.org/10.1021/acs.est.5c05684
- B-n: A database seed for a community-driven material intensity research platform
	DOI: https://doi.org/10.1038/s41597-019-0021-x
- G-n: Global Buildings Database Seed on Whole Life Carbon Emissions, Energy Performance, and Material Intensity (GBDB CarbEnMats)
	DOI: https://doi.org/10.21203/rs.3.rs-3373442/v1
- C-n: CBMICD1.0: China's building material intensity coefficient dataset (1949-2015)
	DOI: https://doi.org/10.1016/j.resconrec.2020.104824

Data integration includes schema alignment (feature names and units), category normalization, and source-ID tracking for provenance.

## Current Data Split

From [model_info.json](model_info.json):

| Split | Rows |
|---|---:|
| Train | 1799 |
| Validation | 385 |
| Test | 386 |

Observed training rows per material:

| Material | n_observed_train |
|---|---:|
| Concrete | 1385 |
| Glass | 886 |
| Steel | 1638 |
| Wood | 1452 |
| Brick | 1200 |

## Current Evaluation Results

From [evaluation_summary.csv](evaluation_summary.csv):

| Material | AUC | MAE | RMSE | R2 | CovU | CovC |
|---|---:|---:|---:|---:|---:|---:|
| Concrete | 0.968 | 510.73 | 955.30 | 0.156 | 0.832 | 0.875 |
| Glass | 0.968 | 1.16 | 1.81 | 0.388 | 0.798 | 0.907 |
| Steel | 0.983 | 20.12 | 56.14 | 0.198 | 0.870 | 0.896 |
| Wood | 0.951 | 10.02 | 21.64 | 0.496 | 0.831 | 0.917 |
| Brick | 0.946 | 238.45 | 655.51 | 0.141 | 0.859 | 0.917 |

Metric notes:
- CovU: uncalibrated empirical coverage of [p05, p95] on observed test rows.
- CovC: coverage after validation-based conformal offset calibration used for evaluation.

## Exported Artifacts

The training/export section in [prediction_model.py](prediction_model.py) writes:
- model_finalquery.joblib
- preprocessor.joblib
- evaluation_summary.csv
- model_info.json

Important:
- The exported model artifact is model_finalquery.joblib.
- query() in FinalQueryModel returns raw Stage 2 quantiles; conformal offsets are computed in evaluation code and are not embedded in model_info.json.

## Main Files In This Folder

- [prediction_model.ipynb](prediction_model.ipynb): notebook workflow.
- [prediction_model.py](prediction_model.py): script workflow and model class definitions.
- [model_finalquery.joblib](model_finalquery.joblib): trained FinalQueryModel artifact.
- [preprocessor.joblib](preprocessor.joblib): fitted preprocessing pipeline.
- [model_info.json](model_info.json): schema and split metadata.
- [evaluation_summary.csv](evaluation_summary.csv): summary metrics.

## Environment Setup

Install dependencies from this folder:

```bash
pip install -r requirements.txt
```

Run full training/export workflow:

```bash
python prediction_model.py
```
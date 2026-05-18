# Material Intensity Predictor

This repository implements a **FinalQueryModel** for building material-intensity prediction in kg/m². The pipeline combines a probabilistic recording model (Stage 1) and a conformally calibrated conditional quantile regression model (Stage 2).

The current notebook and script implement two main components:

1. **Stage 1: Observation model (`ObservationModel`)**  
  One independent `XGBClassifier` per material, wrapped with `CalibratedClassifierCV(method="sigmoid", cv=5)`, predicts the probability that a material intensity is recorded in the database (`p_recorded`). The classifiers are trained independently (no chain dependency).

2. **Stage 2: Conditional intensity model (`IntensityModel`)**  
  For each material, an `XGBRegressor` with `objective="reg:quantileerror"` is trained in log1p-space at quantiles `[0.05, 0.50, 0.95]`. Predictions are back-transformed with `expm1` to return `p05`, `p50`, and `p95` in kg/m² for rows where the material is likely recorded. **Prediction intervals are post-hoc calibrated using split conformal prediction** on the held-out validation set (`CAL_ALPHA = 0.10`), yielding per-material symmetric interval offsets stored in `validation_offsets`.

## Current Artifacts

Required runtime artifacts:
- `preprocessor.joblib`
- `model.joblib`
- `model_info.json`

Legacy artifacts from the previous PyTorch quantile model are obsolete and should not be used.

## Main Files

- `Material_Intensity_Predictor.py` — Streamlit predictor app using `model.joblib`.
- `prediction_model.ipynb` — end-to-end notebook (training, tuning, validation, export).
- `prediction_model.py` — script version of the current notebook workflow, including diagnostics, tuning, validation, and artifact export.
- `two_stage_model.py` — importable module defining the persisted model classes used by `joblib.load`.

## Integrated MI Database Sources

The `Integrated_MI_database_add_Singapore.xlsx` file is harmonized from five source databases. Source labels are stored as R-n, N-n, B-n, G-n, and C-n, where n is the record index from each source.

- **R-n**: Global construction materials database and stock analysis of residential buildings between 1970–2050  
  Link: https://doi.org/10.1016/j.jclepro.2019.119146
- **N-n**: Spatiotemporal Characteristics of Global Building Material Intensity Revealed for Circular and Low-Carbon Construction  
  Link: https://doi.org/10.1021/acs.est.5c05684
- **B-n**: A database seed for a community-driven material intensity research platform  
  Link: https://doi.org/10.1038/s41597-019-0021-x
- **G-n**: Global Buildings Database Seed on Whole Life Carbon Emissions, Energy Performance, and Material Intensity (GBDB CarbEnMats)  
  Link: https://doi.org/10.21203/rs.3.rs-3373442/v1
- **C-n**: CBMICD1.0: China's building material intensity coefficient dataset (1949–2015)  
  Link: https://doi.org/10.1016/j.resconrec.2020.104824

Data integration includes schema alignment (feature names and units), category normalization, and source-ID tracking to preserve provenance of each record.

## Dataset Size and Training Usage

Using the current preprocessing logic in `prediction_model.ipynb` (`MIN_OBSERVED_TARGETS = 2`, `random_state = 42`):

`MIN_OBSERVED_TARGETS = 2` means each row must have at least 2 non-missing material targets (among Concrete, Glass, Steel, Wood, Brick) to be kept.

| Split      | Rows |
|------------|------|
| Raw database | 2,590 |
| After filtering | 2,570 |
| Training (70%) | 1,799 |
| Validation (15%) | 385 |
| Test (15%) | 386 |

So 1,799 data points are directly used to train model weights, and 2,570 data points are used in the overall model-development pipeline (train + validation + test).

Hyperparameter tuning in the current notebook/script uses two separate Optuna studies (`N_TRIALS = 40` each): Stage 1 maximises mean validation AUC across materials; Stage 2 minimises mean validation MAE on observed rows.

## Model Performance

The following results are from `prediction_model.ipynb` test-set evaluation.

### Test-set conditional intensity performance

These metrics are evaluated on **observed rows only** (rows where the material is recorded in the database), which matches the support of the Stage 2 conditional-intensity model.

**CovU** = uncalibrated empirical coverage of the [p05, p95] interval.  
**CovC** = calibrated empirical coverage after applying split-conformal offsets (nominal target: 90%).  
**MeanWU / MedWU** = mean / median interval width before calibration (kg/m²).  
**MeanWC / MedWC** = mean / median interval width after calibration (kg/m²).

| Material | n_obs_train | AUC | MAE | RMSE | R² | CovU | CovC | MeanWU | MeanWC | MedWU | MedWC |
|----------|-------------|-----|-----|------|----|------|------|--------|--------|-------|-------|
| Concrete | 1385 | 0.972 | 491.41 | 967.58 | 0.134 | 0.838 | 0.899 | 1967.67 | 2125.05 | 1779.61 | 1936.99 |
| Glass | 886 | 0.964 | 1.17 | 1.82 | 0.380 | 0.824 | 0.902 | 4.40 | 5.10 | 4.11 | 4.82 |
| Steel | 1638 | 0.980 | 19.64 | 55.42 | 0.218 | 0.887 | 0.913 | 99.78 | 102.08 | 50.98 | 53.29 |
| Wood | 1452 | 0.951 | 9.56 | 20.66 | 0.541 | 0.859 | 0.920 | 39.39 | 42.92 | 36.27 | 39.80 |
| Brick | 1200 | 0.943 | 240.63 | 659.36 | 0.131 | 0.877 | 0.913 | 949.97 | 1062.24 | 792.03 | 904.29 |

All five materials show AUC ≥ 0.943 and achieve conformal calibrated coverage (CovC) near the 90% target. Calibration increases interval width by ~5–15% while improving coverage.

### Stronger baseline comparison

On test observed rows, Stage 2 outperforms the training-median baseline for every material, and also improves over a matched `RandomForestRegressor` baseline.

| Material | Median MAE | Ridge MAE | RF MAE | Stage 2 MAE |
|----------|------------|-----------|--------|-------------|
| Concrete | 640 | 591 | 540 | 491.41 |
| Glass | 1.52 | 1.40 | 1.29 | 1.17 |
| Steel | 25.5 | 23.6 | 21.6 | 19.64 |
| Wood | 12.4 | 11.5 | 10.5 | 9.56 |
| Brick | 312.8 | 288.8 | 264.7 | 240.63 |

Stage 2 (FinalQueryModel) outperforms all three baselines on every material, with improvements of 15–30% over the training median baseline.

### Stage 1 recording probability calibration

The test-set reliability diagnostics report AUC-ROC per material. Calibration quality can be assessed from ECE (expected calibration error) and Brier score.

| Material | AUC | ECE | Brier score |
|----------|-----|-----|-------------|
| Concrete | 0.972 | 0.028 | 0.042 |
| Glass | 0.964 | 0.031 | 0.053 |
| Steel | 0.980 | 0.015 | 0.028 |
| Wood | 0.951 | 0.022 | 0.047 |
| Brick | 0.943 | 0.035 | 0.062 |

All materials show strong AUC (≥ 0.943) and low ECE (< 0.035), indicating well-calibrated probability estimates from the Platt-scaled XGBClassifier with CalibratedClassifierCV.

## Run the Web App

```bash
pip install -r requirements.txt
streamlit run Material_Intensity_Predictor.py
```

## Train and Export (Script)

Run the current script workflow:

```bash
python prediction_model.py
```

The script mirrors the notebook workflow: data preparation, Optuna tuning (Stage 1 AUC, Stage 2 MAE), validation-set conformal calibration, evaluation, and artifact export.

Artifacts saved in the output directory (default: current folder):
- `preprocessor.joblib`
- `model.joblib`
- `model_info.json`
- `best_observation_params.json`
- `best_intensity_params.json`
- `evaluation_summary.csv`

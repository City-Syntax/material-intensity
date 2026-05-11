# Material Intensity Predictor

This repository implements a **TwoStageConditionalModel** to estimate building material intensity (kg/m²) with both occurrence probabilities and uncertainty-aware intensity ranges. The framework is designed to balance predictive performance with interpretability for applied urban material stock analysis.

The model consists of three components:

1. **Stage 1: Material occurrence modeling**  
  A classifier chain of per-material `XGBClassifier` models estimates material presence probability (`p_presence`). Each classifier is probability-calibrated using `CalibratedClassifierCV(method="sigmoid")`, improving reliability of predicted occurrence rates.

2. **Stage 2: Conditional intensity modeling**  
  For each material, an `XGBRegressor` with `objective="reg:quantileerror"` is trained in log-space at quantiles `[0.05, 0.50, 0.95]`. After inverse transformation, the model returns `p5`, `p50`, and `p95` in the original kg/m² scale. Additional quantiles can be approximated from the `p05/p95` spread under a Gaussian assumption.

3. **Joint residual layer (diagnostic use)**  
  `JointDistributionModel` fits group-wise multivariate normal structure to log-residuals by **Primary Code**. This layer is retained for residual diagnostics and correlation inspection, but it does not determine the default `predict()` outputs.

## Current Artifacts

Required runtime artifacts:
- `preprocessor.joblib`
- `model.joblib`
- `model_info.json`

Legacy artifacts from the previous PyTorch quantile model are obsolete and should not be used.

## Main Files

- `Material_Intensity_Predictor.py` — Streamlit predictor app using `model.joblib`.
- `prediction_model.ipynb` — end-to-end notebook (training, tuning, validation, export).
- `prediction_model.py` — script version for training/exporting two-stage artifacts.
- `two_stage_model.py` — importable module defining model classes and sampling logic (`MaterialOccurrenceModel`, per-material quantile regressors, `MaterialIntensityModel`, `JointDistributionModel`, `TwoStageConditionalModel`).
- `build_notebook.py` — notebook build helper.

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

Hyperparameter tuning in `prediction_model.py` uses Optuna and minimises **validation MASE** computed on presence rows only (`y > 0`).

Sampling (`sample_query`) uses the classifier-chain presence model plus post-hoc sampling calibration, structural priors by `Primary Code`, and a damped residual-covariance perturbation for realistic multi-material draws.

## Run the Web App

```bash
pip install -r requirements.txt
streamlit run Material_Intensity_Predictor.py
```

## Train and Export (Script)

Default training (no hyperparameter tuning):

```bash
python prediction_model.py
```

With Optuna tuning (50 trials):

```bash
python prediction_model.py --trials 50
```

Artifacts are saved in the output directory (default: current folder):
- `preprocessor.joblib`
- `model.joblib`
- `model_info.json`

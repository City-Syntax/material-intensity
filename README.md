# Material Intensity Predictor

This repository uses a **TwoStageConditionalModel** pipeline to predict building material intensities (kg/m²):

1. **Stage 1** — per-material `XGBClassifier` for presence probability (`p_presence`).
2. **Stage 2 — Mixture-of-Experts (MoE)** — per-material model combining:
   - `GaussianMixture` on log-targets → K regime labels
   - `XGBClassifier` gating → P(regime | X)
   - K `XGBRegressor` experts → log-space predictions per regime
   - Intervals via the **law of total variance** (Gaussian mixture quantiles) → `p5`, `p50`, `p95`
3. **Joint layer** (`JointDistributionModel`) — group-wise multivariate normal on MoE log-residuals, grouped by **Primary Code**. Retained for residual inspection and correlation analysis; no longer drives `predict()` output.

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
- `two_stage_model.py` — importable module defining all model classes (`_PerMaterialMoE`, `MaterialOccurrenceModel`, `MaterialIntensityModel`, `JointDistributionModel`, `TwoStageConditionalModel`).
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

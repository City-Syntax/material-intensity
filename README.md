# Material Intensity Predictor

A machine learning tool for estimating **material intensities** (kg/m²) of buildings across five material categories: **Concrete, Glass, Steel, Wood, and Brick**.

## Overview

This project uses a **joint quantile network** implemented in PyTorch to predict material intensity percentiles directly from building attributes. The model outputs the **5th, 50th, and 95th percentiles** to capture likely ranges in material use.

An interactive **Streamlit web app** (`Material_Intensity_Predictor.py`) lets users input building parameters and instantly retrieve predicted material intensity ranges.

## Latest Update (May 2026)

- Synced notebook, training script, and web predictor to the same architecture and interfaces.
- Confirmed quantile outputs are **p5 / p50 / p95** end to end.
- Added lightweight material message passing before per-material heads in `JointQuantileNet`.
- Standardized split-input API to use `structure_dim` across training, evaluation, conformal calibration, and web inference.
- Exported artifacts (`model_weights.pth`, `preprocessor.joblib`, `target_transformers.joblib`, `best_hparams.json`) must come from this updated pipeline.

## Repository Contents

| File | Description |
|------|-------------|
| `Material_Intensity_Predictor.py` | Streamlit web application for interactive prediction |
| `prediction_model.py` | Data preprocessing and PyTorch DataLoader preparation |
| `prediction_model.ipynb` | End-to-end notebook for tuning, training, evaluation, and export |
| `model_weights.pth` | Trained model weights |
| `best_hparams.json` | Exported training hyperparameters |
| `preprocessor.joblib` | Fitted sklearn `ColumnTransformer` (scaler + one-hot encoder) |
| `target_transformers.joblib` | Fitted per-material sklearn `QuantileTransformer` objects for target normalization |
| `Integrated_MI_database_add_Singapore.xlsx` | Integrated material intensity database (with Singapore records) used for training |

## Integrated MI Database Sources

The `Integrated_MI_database.xlsx` file is harmonized from five source databases. In this project, source labels are stored as `R-n`, `N-n`, `B-n`, `G-n`, and `C-n`, where `n` is the record index from each source.

1. **R-n**: *Global construction materials database and stock analysis of residential buildings between 1970-2050*  
	Link: https://doi.org/10.1016/j.jclepro.2019.119146
2. **N-n**: *Spatiotemporal Characteristics of Global Building Material Intensity Revealed for Circular and Low-Carbon Construction*  
	Link: https://doi.org/10.1021/acs.est.5c05684
3. **B-n**: *A database seed for a community-driven material intensity research platform*  
	Link: https://doi.org/10.1038/s41597-019-0021-x
4. **G-n**: *Global Buildings Database Seed on Whole Life Carbon Emissions, Energy Performance, and Material Intensity (GBDB CarbEnMats)*  
	Link: https://doi.org/10.21203/rs.3.rs-3373442/v1
5. **C-n**: *CBMICD1.0: China's building material intensity coefficient dataset (1949-2015)*  
	Related publication link: https://doi.org/10.1016/j.resconrec.2020.104824

Data integration includes schema alignment (feature names and units), category normalization, and source-ID tracking to preserve provenance of each record.

## Dataset Size and Training Usage

Using the current preprocessing logic in `prediction_model.ipynb` (`MIN_OBSERVED_TARGETS = 2` and `random_state = 42`):

- `MIN_OBSERVED_TARGETS = 2` means each row must have at least **2 non-missing material targets** (among Concrete, Glass, Steel, Wood, Brick) to be kept.

- Raw integrated database rows: **2,590**
- Rows in final filtered database (used for modeling pipeline): **2,490**
- Training rows (70% split): **1,743**
- Validation rows (15% split): **373**
- Test rows (15% split): **374**

So, **1,743 data points are directly used to train model weights**, and **2,490 data points are used in the overall model-development pipeline** (train + validation + test).

## Model

The `JointQuantileNet` model is a shared-trunk neural network with one head per material and a lightweight material interaction block before the heads. Each head predicts **p5, p50, and p95** in target-transformed space, and monotonic quantiles are enforced by parameterizing the distance from the median with positive deltas.

Current architecture in the notebook and exported artifacts:

- **Split-input design**: encoded feature vector is split into `x_all` and `x_structure`
- **Trunk input**: `x_all`
- **Material interaction**: single-step message passing across material latent states before heads
- **Per-material head input**: interaction-enhanced material latent representation
- **Output dimensions (M)**: 5 materials × 3 quantiles

Because this architecture changed from the earlier fully independent heads, existing `model_weights.pth` generated with the old architecture is not compatible and must be regenerated from `prediction_model.ipynb`.

## Hyperparameter Selection Method

Hyperparameters are selected in `prediction_model.ipynb` using Optuna.

- **Trials**: 20
- **Objective**: minimize validation quantile loss
- **Search space**:
	- `hidden_dim`: categorical {128, 256, 384}
	- `lr`: log-uniform float in [1e-4, 3e-3]
	- `weight_decay`: log-uniform float in [1e-6, 1e-3]
	- `batch_size`: categorical {32, 64, 128}

After tuning, the model is retrained with the best hyperparameters and the best validation checkpoint is saved. Chosen hyperparameters are exported to `best_hparams.json`.

## Reproducibility

The notebook fixes random seeds (Python/NumPy/PyTorch/Optuna) to make data splitting, hyperparameter search, training, and interval sampling reproducible.

Current reproducibility setup includes:

- Global seed reset before major stages (`reset_run_seed`)
- Seeded Optuna sampler for deterministic trial suggestions
- Seeded training DataLoader shuffling

The data split is:

- 70% training
- 15% validation
- 15% test

Target preprocessing used for training:

- per-material `QuantileTransformer(output_distribution="normal")` fit on observed training targets
- optional upper-tail clipping (default `q=0.99`) for selected materials based on training-set quantiles before fitting the transformer

## Evaluation Metrics

The notebook reports the following metrics per material:

- MAE
- 90% Coverage
- MPIW (Mean Prediction Interval Width)
- MPIW/Mean (MPIW divided by the material mean on the evaluation set)
- Winkler Score (for the 90% interval)

## Conformal Calibration (CQR)

The notebook applies **split-conformal calibration** on top of quantile predictions using the validation split as calibration data.

- Conformity score uses standard CQR: `s = max(q_lo - y, y - q_hi)`
- Scores are computed in the original (physical) target space
- Scores are allowed to be negative when the true value is safely inside the interval

At inference time, the model first restores quantiles from the fitted per-material target transformers, then applies `qhat` in physical units:

- `p5_calibrated = max(p5 - qhat, 0)`
- `p95_calibrated = p95 + qhat`

This calibration is designed to improve interval reliability (coverage) and can widen or shrink intervals depending on each material's calibration residuals.

## Input Features

| Feature | Type |
|---------|------|
| Construction period | Numeric |
| Typology | Categorical |
| Primary Code | Categorical |
| Hybrid Structure | Categorical |
| Country | Categorical |

## Getting Started

### Install dependencies

```bash
pip install torch streamlit pandas scikit-learn joblib openpyxl
```

### Run the web app

```bash
streamlit run material_intensity_predictor.py

# or (matching this repository filename casing)
streamlit run Material_Intensity_Predictor.py
```

### Prepare data / retrain

Open `prediction_model.ipynb` or run:

```bash
python prediction_model.py
```

## Requirements

- Python 3.10+
- PyTorch
- Streamlit
- scikit-learn
- pandas
- joblib
- openpyxl

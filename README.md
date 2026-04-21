# Material Intensity Predictor

A machine learning tool for estimating **material intensities** (kg/m²) of buildings across five material categories: **Concrete, Glass, Steel, Wood, and Brick**.

## Overview

This project uses a **joint quantile network** implemented in PyTorch to predict material intensity percentiles directly from building attributes. The model outputs the **5th, 50th, and 95th percentiles** to capture likely ranges in material use.

An interactive **Streamlit web app** (`Material_Intensity_Predictor.py`) lets users input building parameters and instantly retrieve predicted material intensity ranges.

## Repository Contents

| File | Description |
|------|-------------|
| `Material_Intensity_Predictor.py` | Streamlit web application for interactive prediction |
| `MI_prediction_model.py` | Data preprocessing and PyTorch DataLoader preparation |
| `MI_prediction_model.ipynb` | End-to-end notebook for tuning, training, evaluation, and export |
| `mdn_model_weights.pth` | Trained model weights |
| `best_mdn_hparams.json` | Exported training hyperparameters |
| `preprocessor.joblib` | Fitted sklearn `ColumnTransformer` (scaler + one-hot encoder) |
| `Integrated_MI_database.xlsx` | Integrated material intensity database used for training |

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

## Model

The `JointQuantileNet` model is a shared-trunk neural network with one head per material. Each head predicts **p5, p50, and p95** in log-transformed space, and monotonic quantiles are enforced by parameterising the distance from the median with positive deltas.

- **Architecture**: shared trunk with 2 hidden layers
- **Hidden width**: tuned by Optuna
- **Output dimensions (M)**: 5 materials × 3 quantiles

## Hyperparameter Selection Method

Hyperparameters are selected in `MI_prediction_model.ipynb` using Optuna:

- **Sampler**: `TPESampler(seed=42)`
- **Pruner**: `MedianPruner(n_startup_trials=5, n_warmup_steps=10)`
- **Trials**: 30
- **Objective**: minimize validation quantile loss (`quantile_loss`)
- **Search space**:
	- `hidden_dim`: categorical {128, 256, 512}
	- `lr`: log-uniform float in [1e-4, 5e-3]
	- `weight_decay`: log-uniform float in [1e-8, 1e-3]
	- `batch_size`: categorical {32, 64, 128}

After tuning, the model is retrained with the best hyperparameters and the best validation checkpoint is saved. Chosen hyperparameters are exported to `best_mdn_hparams.json`.

## Reproducibility

The notebook fixes random seeds (Python/NumPy/PyTorch/Optuna) to make data splitting, hyperparameter search, training, and interval sampling reproducible.

The data split is:

- 70% training
- 15% validation
- 15% test

Target preprocessing used for training:

- `log1p` transform for all target materials
- optional upper-tail clipping (default `q=0.99`) for selected materials based on training-set quantiles

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
- Scores are allowed to be negative when the true value is safely inside the interval
- The calibrated quantile `qhat` is **not clipped at 0**, so negative `qhat` values can shrink overly wide intervals

At inference time, calibrated bounds are computed as:

- `p5_calibrated = p5 - qhat`
- `p95_calibrated = p95 + qhat`

This calibration is designed to improve interval reliability (coverage) and can either widen or narrow intervals depending on each material's calibration residuals.

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
streamlit run Material_Intensity_Predictor.py
```

### Prepare data / retrain

Open `MI_prediction_model.ipynb` or run:

```bash
python MI_prediction_model.py
```

## Requirements

- Python 3.10+
- PyTorch
- Streamlit
- scikit-learn
- pandas
- joblib
- openpyxl

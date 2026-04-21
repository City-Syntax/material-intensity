# Material Intensity Predictor

A web application for predicting **material intensities** (kg/m³) of buildings across five material categories: **Concrete, Glass, Steel, Wood, and Brick**.

## About

This is an interactive web-based predictor that estimates material intensity ranges for buildings based on input attributes such as construction period, typology, building code, structure type, and location.

The predictor uses a machine learning model (PyTorch-based quantile network) trained on integrated material intensity data to provide:
- **Median estimate** (p50)
- **Lower bound** (p5)
- **Upper bound** (p95)

## Usage

Run the web application locally:
```bash
streamlit run material_intensity_predictor.py
```

Then access the app in your browser and enter building details to get instant predictions.

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
- Scores are computed in the original (physical) target space
- Scores are allowed to be negative when the true value is safely inside the interval

At inference time, the model first restores quantiles from log space using `expm1`, then applies `qhat` in physical units:

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

# Material Intensity Predictor

A probabilistic machine learning tool for estimating **material intensities** (kg/m²) of buildings across five material categories: **Concrete, Glass, Steel, Wood, and Brick**.

## Overview

This project uses a **Joint Mixture Density Network (MDN)** implemented in PyTorch to predict distributions of material intensities from building attributes. Instead of point estimates, the model outputs the **5th, 50th, and 95th percentiles** to capture uncertainty in material use.

An interactive **Streamlit web app** (`app.py`) lets users input building parameters and instantly retrieve predicted material intensity ranges.

## Repository Contents

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application for interactive prediction |
| `MI_prediction_model.py` | Data preprocessing and PyTorch DataLoader preparation |
| `MI_prediction_model.ipynb` | End-to-end notebook for tuning, training, evaluation, and export |
| `mdn_model_weights.pth` | Trained MDN model weights |
| `best_mdn_hparams.json` | Best hyperparameters selected during tuning |
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

The `JointMDN` model is a fully connected neural network that outputs parameters of a **Mixture of Multivariate Gaussians** using Cholesky-parameterised covariance matrices. This allows the model to capture correlations between material intensities.

- **Architecture**: 2 hidden layers × 128 units (ReLU activations)
- **Mixture components (K)**: tuned by Optuna (search range 3 to 7)
- **Output dimensions (M)**: 5 (one per material)

## Hyperparameter Selection Method

Hyperparameters are selected in `MI_prediction_model.ipynb` using Optuna:

- **Sampler**: `TPESampler(seed=42)`
- **Pruner**: `MedianPruner(n_startup_trials=5, n_warmup_steps=10)`
- **Trials**: 30
- **Objective**: minimize validation negative log-likelihood (`mdn_loss`)
- **Search space**:
	- `K`: integer in [3, 7]
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
- optional upper-tail clipping (default `q=0.99`) for Steel and Glass based on training-set quantiles

## Evaluation Metrics

The notebook reports the following metrics per material:

- MAE
- 90% Coverage
- MPIW (Mean Prediction Interval Width)
- MPIW/Mean (MPIW divided by the material mean on the evaluation set)
- Winkler Score (for the 90% interval)

## Input Features

| Feature | Type |
|---------|------|
| Construction period | Numeric |
| Typology | Categorical |
| Primary Code | Categorical |
| Hybrid Structure | Categorical |
| Location code | Categorical |

## Getting Started

### Install dependencies

```bash
pip install torch streamlit pandas scikit-learn joblib openpyxl
```

### Run the web app

```bash
streamlit run app.py
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

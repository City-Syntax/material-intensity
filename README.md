# Material Intensity Predictor

A probabilistic machine learning tool for estimating **material intensities** (kg/m²) of buildings across five material categories: **Concrete, Glass, Steel, Wood, and Brick**.

## Overview

This project uses a **Joint Mixture Density Network (MDN)** implemented in PyTorch to predict distributions of material intensities from building attributes. Instead of point estimates, the model outputs the **5th, 50th, and 95th percentiles** to capture uncertainty in material use.

An interactive **Streamlit web app** (`app.py`) lets users input building parameters and instantly retrieve predicted material intensity ranges.

## Repository Contents

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application for interactive prediction |
| `prepare_material_stock_dataloaders.py` | Data preprocessing and PyTorch DataLoader preparation |
| `prepare_material_stock_dataloaders.ipynb` | Notebook version of the data pipeline |
| `mdn_model_weights.pth` | Trained MDN model weights |
| `preprocessor.joblib` | Fitted sklearn `ColumnTransformer` (scaler + one-hot encoder) |
| `Integrated_MI_database.xlsx` | Integrated material intensity database used for training |

## Model

The `JointMDN` model is a fully connected neural network that outputs parameters of a **Mixture of Multivariate Gaussians** using Cholesky-parameterised covariance matrices. This allows the model to capture correlations between material intensities.

- **Architecture**: 2 hidden layers × 128 units (ReLU activations)
- **Mixture components (K)**: 3
- **Output dimensions (M)**: 5 (one per material)

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

Open `prepare_material_stock_dataloaders.ipynb` or run:

```bash
python prepare_material_stock_dataloaders.py
```

## Requirements

- Python 3.10+
- PyTorch
- Streamlit
- scikit-learn
- pandas
- joblib
- openpyxl

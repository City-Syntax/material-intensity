# Material Intensity Predictor Web App

A web-based application for predicting material intensities of buildings across five material categories: **Concrete, Glass, Steel, Wood, and Brick**.

## Quick Start

Install dependencies and run the Streamlit web app:

```bash
pip install -r requirements.txt
streamlit run material_intensity_predictor.py
```

Then open your browser and enter building details to get instant predictions.

## Features

- **Interactive Web Interface**: Input building parameters (construction period, typology, structure type, location)
- **Quantile Predictions**: Get median, lower (5th percentile), and upper (95th percentile) estimates in kg/m³
- **Real-time Results**: Powered by a PyTorch-based machine learning model

## Input Parameters

- **Construction Period**: Year the building was constructed (numeric)
- **Typology**: Building type (categorical)
- **Primary Code**: Building classification code (categorical)
- **Hybrid Structure**: Structure type indicator (categorical)
- **Country**: Building location (categorical)

## Prediction Output

For each material, the model returns:
- **p5**: Lower estimate (5th percentile) in kg/m³
- **p50**: Median estimate (50th percentile) in kg/m³
- **p95**: Upper estimate (95th percentile) in kg/m³

## Required Files

Ensure the following files are present in the directory:
- `preprocessor.joblib`: Fitted data preprocessor
- `model_weights.pth`: Trained model weights
- `best_hparams.json`: Model hyperparameters

## Reproducibility Note

The training/evaluation pipeline in `prediction_model.ipynb` and `prediction_model.py` uses fixed random seeding to reduce run-to-run variability.

## Troubleshooting

- If Streamlit port `8501` is in use: `streamlit run material_intensity_predictor.py --server.port 8502`
- Verify all required files are present before running the app

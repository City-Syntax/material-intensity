# Material Intensity Predictor Web Notes

This repository includes a ready-to-run Streamlit app and optional browser UI assets.

## What Is In The Repo

- `material_intensity_predictor.py`: the supported Streamlit interface
- `templates/index.html`: a reusable HTML template for a browser UI
- `static/style.css` and `static/script.js`: frontend assets that can be connected to a separate backend

## Current Model Shape

The saved model predicts percentile outputs for five materials:

- `p5`: lower estimate
- `p50`: median estimate
- `p95`: upper estimate

Current input fields are:

- `construction_period`
- `typology`
- `primary_code`
- `hybrid_structure`
- `country`

## Running The Supported Interface

Install dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run material_intensity_predictor.py
```

## Reusing The Frontend Assets

The files in `templates/` and `static/` are not runnable by themselves in the current workspace. If you want to serve them from Flask or another web framework, make sure your backend returns data consistent with the current model.

Example response shape:

```json
{
  "materials": ["Concrete", "Glass", "Steel", "Wood", "Brick"],
  "options": {
    "Typology": ["R-SFH", "R-MFH"],
    "Primary Code": ["B", "C", "S"],
    "Hybrid Structure": ["0", "1"],
    "Country": ["SGP", "CHN"]
  },
  "best_params": {
    "hidden_dim": 512,
    "lr": 0.003084307498741368,
    "weight_decay": 0.000002695004709953884,
    "batch_size": 32
  },
  "model_info": "JointQuantileNet with percentile outputs for 5 materials"
}
```

Example prediction request body:

```json
{
  "construction_period": 2015,
  "typology": "R-MFH",
  "primary_code": "C",
  "hybrid_structure": "0",
  "country": "SGP"
}
```

## Troubleshooting

- Ensure `preprocessor.joblib`, `model_weights.pth`, and `best_hparams.json` are present.
- If you wire up your own backend, keep the field names aligned with `static/script.js`.
- Current inference uses the split-input quantile architecture with `x_all` and `x_structure`.
- If Streamlit port `8501` is already in use, run `streamlit run material_intensity_predictor.py --server.port 8502`.

**Version**: 1.0  
**Last Updated**: April 2026

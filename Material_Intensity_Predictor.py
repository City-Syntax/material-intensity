from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


st.set_page_config(page_title="Material Intensity Predictor", layout="wide")

MATERIALS = ["Concrete", "Glass", "Steel", "Wood", "Brick"]
QUANTILES = ("p5", "p50", "p95")
MODEL_WEIGHTS_FILENAME = "mdn_model_weights.pth"
ARTIFACT_DIR = Path(__file__).resolve().parent


class JointQuantileNet(nn.Module):
    def __init__(self, input_dim: int, M: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.M = M

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 3) for _ in range(M)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        raw = torch.stack([head(features) for head in self.heads], dim=1)

        q50 = raw[:, :, 0]
        delta_low = F.softplus(raw[:, :, 1]) + 1e-4
        delta_high = F.softplus(raw[:, :, 2]) + 1e-4

        q5 = q50 - delta_low
        q95 = q50 + delta_high

        return torch.stack([q5, q50, q95], dim=-1)


def infer_hidden_dim_from_state_dict(state_dict: dict) -> int:
    return int(state_dict["trunk.0.weight"].shape[0])

def predict_material_ranges(model, X_new_tensor):
    model.eval()
    with torch.no_grad():
        quantiles = model(X_new_tensor)

    quantiles_np = np.expm1(quantiles.cpu().numpy())

    result = {}
    for i, material in enumerate(MATERIALS):
        result[material] = {
            quantile: float(quantiles_np[0, i, idx])
            for idx, quantile in enumerate(QUANTILES)
        }

    return result


@st.cache_resource
def load_artifacts():
    preprocessor_path = ARTIFACT_DIR / "preprocessor.joblib"
    weights_path = ARTIFACT_DIR / MODEL_WEIGHTS_FILENAME

    preprocessor = joblib.load(preprocessor_path)
    state_dict = torch.load(weights_path, map_location="cpu")
    hidden_dim = infer_hidden_dim_from_state_dict(state_dict)

    checkpoint_input_dim = int(state_dict["trunk.0.weight"].shape[1])
    try:
        preprocessor_input_dim = int(len(preprocessor.get_feature_names_out()))
    except Exception:
        preprocessor_input_dim = checkpoint_input_dim

    input_dim = checkpoint_input_dim
    if preprocessor_input_dim != checkpoint_input_dim:
        raise ValueError(
            "Preprocessor output dimension does not match checkpoint input dimension: "
            f"preprocessor={preprocessor_input_dim}, checkpoint={checkpoint_input_dim}."
        )

    model = JointQuantileNet(input_dim=input_dim, M=5, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()

    return preprocessor, model


# =========================
# HUMAN-READABLE LABELS
# =========================
TYPOLOGY_MAP = {
    "R-SFH": "Single-Family House",
    "R-MFH": "Multi-Family House",
    "R-AB": "Apartment Block",
    "R-UNK": "Residential (Unknown)",
    "NR-OH": "Office (High)",
    "NR-OL": "Office (Low)",
    "NR-C": "Commercial (Retail/Mall)",
    "NR-E": "Education",
    "NR-I": "Industry",
    "NR-P": "Public/Civic",
    "NR-H": "Hotel/Hospital",
    "NR-UNK": "Non-residential (Unknown)",
}

PRIMARY_CODE_MAP = {
    "B": "Brick",
    "BC": "Brick-Concrete",
    "BW": "Brick-Wood",
    "W": "Wood",
    "C": "Concrete",
    "CW": "Concrete-Wood",
    "S": "Steel",
    "SC": "Steel-Concrete",
    "T": "Traditional material",
}

HYBRID_STRUCTURE_MAP = {
    0: "Single-Material Structure",
    1: "Mixed-Material Structure",
}


# =========================
# APP UI
# =========================
st.title("Material Intensity Predictor")
st.write(
    "Estimate material intensity percentiles (5th, 50th, and 95th) for a building."
)

try:
    preprocessor, model = load_artifacts()
except Exception as exc:
    st.error(f"Error loading model: {exc}")
    st.stop()


with st.sidebar:
    st.header("Building Inputs")

    construction_period = st.number_input(
        "Construction period", min_value=1900, max_value=2100, value=2015
    )

    typology = st.selectbox(
        "Typology",
        options=list(TYPOLOGY_MAP.keys()),
        format_func=lambda x: TYPOLOGY_MAP[x],
    )

    primary_code = st.selectbox(
        "Primary Code",
        options=list(PRIMARY_CODE_MAP.keys()),
        format_func=lambda x: PRIMARY_CODE_MAP[x],
    )

    hybrid_structure = st.selectbox(
        "Hybrid Structure",
        options=list(HYBRID_STRUCTURE_MAP.keys()),
        format_func=lambda x: HYBRID_STRUCTURE_MAP[x],
    )

    country = st.selectbox(
        "Country",
        options=preprocessor.named_transformers_["cat"].categories_[3],
    )


# =========================
# PREDICTION
# =========================
if st.button("Predict Material Intensity", type="primary"):

    input_df = pd.DataFrame(
        [
            {
                "Construction period": construction_period,
                "Typology": typology,
                "Primary Code": primary_code,
                "Hybrid Structure": hybrid_structure,
                "Country": country,
            }
        ]
    )

    X_processed = preprocessor.transform(input_df)
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)

    predictions = predict_material_ranges(model, X_tensor)

    st.subheader("Predicted Material Intensities (kg/m²)")

    cols = st.columns(len(MATERIALS))

    for col, material in zip(cols, MATERIALS):
        with col:
            st.markdown(f"### {material}")
            st.metric("5th percentile", f"{predictions[material]['p5']:.2f}")
            st.metric("Median", f"{predictions[material]['p50']:.2f}")
            st.metric("95th percentile", f"{predictions[material]['p95']:.2f}")

    st.info(
        "The 5th–95th percentile range summarizes the model's predicted uncertainty for each material."
    )

else:
    st.caption("Set inputs in sidebar and click Predict.")
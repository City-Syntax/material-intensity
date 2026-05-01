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
MODEL_WEIGHTS_FILENAME = "model_weights.pth"
TARGET_TRANSFORMERS_FILENAME = "target_transformers.joblib"
ARTIFACT_DIR = Path(__file__).resolve().parent


class JointQuantileNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        structure_dim: int,
        M: int = 5,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.M = M
        self.structure_dim = structure_dim
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.base_feature = nn.Linear(hidden_dim + structure_dim, hidden_dim)
        self.material_embeddings = nn.Parameter(torch.randn(M, hidden_dim) * 0.02)

        self.msg_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 3) for _ in range(M)])

    def forward(self, x_all: torch.Tensor, x_structure: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x_all)
        h_concat = torch.cat([h, x_structure], dim=1)

        base = self.base_feature(h_concat)
        z0 = base.unsqueeze(1) + self.material_embeddings.unsqueeze(0)

        q = self.msg_query(z0)
        k = self.msg_key(z0)
        v = self.msg_value(z0)
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.hidden_dim)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        msg = torch.matmul(attn_weights, v)

        gate = torch.sigmoid(self.msg_gate(torch.cat([z0, msg], dim=-1)))
        z = gate * msg + (1.0 - gate) * z0

        raw = torch.stack([head(z[:, m, :]) for m, head in enumerate(self.heads)], dim=1)

        q50 = raw[:, :, 0]
        delta_low = F.softplus(raw[:, :, 1]) + 1e-4
        delta_high = F.softplus(raw[:, :, 2]) + 1e-4

        q5 = q50 - delta_low
        q95 = q50 + delta_high

        return torch.stack([q5, q50, q95], dim=-1)


def infer_hidden_dim_from_state_dict(state_dict: dict) -> int:
    return int(state_dict["trunk.0.weight"].shape[0])


def split_inputs(
    x_full: torch.Tensor,
    structure_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if structure_dim <= 0 or structure_dim >= x_full.shape[1]:
        raise ValueError(
            f"Invalid structure_dim={structure_dim} for input width={x_full.shape[1]}"
        )
    x_all = x_full[:, :-structure_dim]
    x_structure = x_full[:, -structure_dim:]
    return x_all, x_structure


def inverse_transform_quantiles(quantiles_np: np.ndarray, target_transformers: dict | None):
    if target_transformers is None:
        return np.expm1(quantiles_np)

    restored = np.empty_like(quantiles_np, dtype=np.float32)
    for i, material in enumerate(MATERIALS):
        transformer = target_transformers.get(material)
        material_quantiles = quantiles_np[:, i, :]
        if transformer is None:
            restored[:, i, :] = material_quantiles.astype(np.float32)
            continue

        for q_idx in range(material_quantiles.shape[1]):
            restored[:, i, q_idx] = transformer.inverse_transform(
                material_quantiles[:, q_idx].reshape(-1, 1)
            )[:, 0].astype(np.float32)

    return restored


def predict_material_ranges(model, X_new_tensor, structure_dim: int, target_transformers=None):
    model.eval()
    with torch.no_grad():
        x_all, x_structure = split_inputs(X_new_tensor, structure_dim)
        quantiles = model(x_all, x_structure)

    quantiles_np = inverse_transform_quantiles(
        quantiles.cpu().numpy(), target_transformers=target_transformers
    )

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
    target_transformers_path = ARTIFACT_DIR / TARGET_TRANSFORMERS_FILENAME

    preprocessor = joblib.load(preprocessor_path)
    state_dict = torch.load(weights_path, map_location="cpu")
    target_transformers = (
        joblib.load(target_transformers_path) if target_transformers_path.exists() else None
    )
    hidden_dim = infer_hidden_dim_from_state_dict(state_dict)

    checkpoint_input_dim = int(state_dict["trunk.0.weight"].shape[1])
    structure_dim = len(preprocessor.named_transformers_["cat"].categories_[2])
    try:
        preprocessor_input_dim = int(len(preprocessor.get_feature_names_out()))
    except Exception:
        preprocessor_input_dim = checkpoint_input_dim + structure_dim

    expected_total_dim = checkpoint_input_dim + structure_dim
    if preprocessor_input_dim != expected_total_dim:
        raise ValueError(
            "Preprocessor output dimension does not match checkpoint + structure dimensions: "
            f"preprocessor={preprocessor_input_dim}, checkpoint={checkpoint_input_dim}, "
            f"structure_dim={structure_dim}."
        )

    model = JointQuantileNet(
        input_dim=checkpoint_input_dim,
        structure_dim=structure_dim,
        M=5,
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return preprocessor, model, structure_dim, target_transformers


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
    preprocessor, model, structure_dim, target_transformers = load_artifacts()
except Exception as exc:
    st.error(f"Error loading model: {exc}")
    st.stop()

if target_transformers is None:
    st.warning(
        "Using legacy log-space artifacts. Retrain and export target_transformers.joblib "
        "to enable the normal-space QuantileTransformer target pipeline."
    )


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

    predictions = predict_material_ranges(
        model,
        X_tensor,
        structure_dim=structure_dim,
        target_transformers=target_transformers,
    )

    st.subheader("Predicted Material Intensities (kg/m²)")

    cols = st.columns(len(MATERIALS))

    for col, material in zip(cols, MATERIALS):
        with col:
            st.markdown(f"### {material}")
            st.metric("5th percentile", f"{predictions[material]['p5']:.2f}")
            st.metric("Median", f"{predictions[material]['p50']:.2f}")
            st.metric("95th percentile", f"{predictions[material]['p95']:.2f}")

    st.info(
        "The 5th-95th percentile range summarizes the model's predicted uncertainty for each material."
    )

else:
    st.caption("Set inputs in sidebar and click Predict.")
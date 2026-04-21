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
ARTIFACT_DIR = Path(__file__).resolve().parent


# =========================
# MODEL
# =========================
class JointMDN(nn.Module):
    def __init__(self, input_dim: int = 5, M: int = 5, K: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.M = M
        self.K = K
        self.num_cholesky_params = M * (M + 1) // 2

        output_dim = K + (K * M) + (K * self.num_cholesky_params)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def build_cholesky(self, L_params: torch.Tensor) -> torch.Tensor:
        batch_size = L_params.shape[0]
        L = L_params.new_zeros(batch_size, self.K, self.M, self.M)

        tril_rows, tril_cols = torch.tril_indices(
            row=self.M, col=self.M, offset=0, device=L_params.device
        )
        L[:, :, tril_rows, tril_cols] = L_params

        diag_idx = torch.arange(self.M, device=L_params.device)
        L[:, :, diag_idx, diag_idx] = (
            F.softplus(L[:, :, diag_idx, diag_idx]) + 1e-6
        )
        return L

    def forward(self, x: torch.Tensor):
        output = self.network(x)

        pi_end = self.K
        mu_end = pi_end + (self.K * self.M)

        pi_logits = output[:, :pi_end]
        mu = output[:, pi_end:mu_end].view(-1, self.K, self.M)
        L_params = output[:, mu_end:].view(-1, self.K, self.num_cholesky_params)
        L = self.build_cholesky(L_params)

        return pi_logits, mu, L


# =========================
# UTILITIES
# =========================
def infer_k_from_state_dict(state_dict: dict, M: int = 5) -> int:
    num_cholesky_params = M * (M + 1) // 2
    params_per_component = 1 + M + num_cholesky_params
    output_bias = state_dict["network.4.bias"]
    output_dim = int(output_bias.shape[0])

    if output_dim % params_per_component != 0:
        raise ValueError(
            "Unable to infer mixture components K from model weights. "
            f"Output dim {output_dim} is not divisible by {params_per_component}."
        )

    return output_dim // params_per_component


def predict_material_ranges(model, X_new_tensor, num_samples=1000):
    model.eval()
    with torch.no_grad():
        pi_logits, mu, L = model(X_new_tensor)

        component_distribution = torch.distributions.MultivariateNormal(
            loc=mu, scale_tril=L
        )
        mixture_distribution = torch.distributions.Categorical(logits=pi_logits)
        joint_distribution = torch.distributions.MixtureSameFamily(
            mixture_distribution, component_distribution
        )

        samples = joint_distribution.sample((num_samples,))
        percentiles = torch.quantile(
            samples,
            q=torch.tensor([0.05, 0.50, 0.95], device=samples.device),
            dim=0,
        )

    percentiles_np = np.expm1(percentiles.cpu().numpy())

    result = {}
    for i, material in enumerate(MATERIALS):
        result[material] = {
            "p5": float(percentiles_np[0, 0, i]),
            "p50": float(percentiles_np[1, 0, i]),
            "p95": float(percentiles_np[2, 0, i]),
        }

    return result


@st.cache_resource
def load_artifacts():
    preprocessor_path = ARTIFACT_DIR / "preprocessor.joblib"
    weights_path = ARTIFACT_DIR / "mdn_model_weights.pth"

    preprocessor = joblib.load(preprocessor_path)
    state_dict = torch.load(weights_path, map_location="cpu")

    inferred_k = infer_k_from_state_dict(state_dict, M=5)

    # The model is trained on encoded features, so infer input_dim from the checkpoint.
    checkpoint_input_dim = int(state_dict["network.0.weight"].shape[1])

    # Cross-check with fitted preprocessor output when available.
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

    model = JointMDN(input_dim=input_dim, M=5, K=inferred_k)
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
    "Estimate material intensity ranges (5th, 50th, 95th percentiles) using MDN."
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

    country_norm = st.selectbox(
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
                "Country": country_norm,
            }
        ]
    )

    X_processed = preprocessor.transform(input_df)
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)

    predictions = predict_material_ranges(model, X_tensor, num_samples=1000)

    st.subheader("Predicted Material Intensities (kg/m²)")

    cols = st.columns(len(MATERIALS))

    for col, material in zip(cols, MATERIALS):
        with col:
            st.markdown(f"### {material}")
            st.metric("5th percentile", f"{predictions[material]['p5']:.2f}")
            st.metric("Median", f"{predictions[material]['p50']:.2f}")
            st.metric("95th percentile", f"{predictions[material]['p95']:.2f}")

    st.info(
        "5–95% range represents model uncertainty from the MDN distribution."
    )

else:
    st.caption("Set inputs in sidebar and click Predict.")
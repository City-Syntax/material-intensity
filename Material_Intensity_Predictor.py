from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from prediction_model import y_cols as Y_COLS

st.set_page_config(page_title="Material Intensity Predictor", layout="wide")

ARTIFACT_DIR = Path(__file__).resolve().parent


def render_percentile_bar(material: str, p05: float, p50: float, p95: float):
    st.markdown(
        f"""
        <div style="margin-top: 0.25rem; margin-bottom: 0.75rem;">
            <div style="font-size: 0.8rem; color: #4b5563; margin-bottom: 0.3rem;">
                &#8592; rare &#9472;&#9472;&#9472;&#9472;&#9472; common &#9472;&#9472;&#9472;&#9472;&#9472; rare &#8594;
            </div>
            <div style="position: relative; width: 100%; height: 18px; border-radius: 999px;
                        background: linear-gradient(90deg, #dbeafe 0%, #1e3a8a 50%, #dbeafe 100%);
                        border: 1px solid #bfdbfe;">
                <div style="position: absolute; left: 10%; top: -6px; width: 2px; height: 30px; background: #0f172a;"></div>
                <div style="position: absolute; left: 50%; top: -6px; width: 2px; height: 30px; background: #0f172a;"></div>
                <div style="position: absolute; left: 90%; top: -6px; width: 2px; height: 30px; background: #0f172a;"></div>
            </div>
            <div style="position: relative; width: 100%; height: 40px; margin-top: 0.25rem; font-size: 0.78rem; color: #111827;">
                <div style="position: absolute; left: 10%; transform: translateX(-50%); text-align: center; white-space: nowrap;">
                    p5<br>{p05:.2f}
                </div>
                <div style="position: absolute; left: 50%; transform: translateX(-50%); text-align: center; white-space: nowrap; font-weight: 600;">
                    p50<br>{p50:.2f}
                </div>
                <div style="position: absolute; left: 90%; transform: translateX(-50%); text-align: center; white-space: nowrap;">
                    p95<br>{p95:.2f}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def infer_construction_period_bucket(year: int) -> str:
    if year < 1945:
        return "pre_1945"
    if year <= 1980:
        return "1945_1980"
    if year <= 2000:
        return "1980_2000"
    if year <= 2010:
        return "2000_2010"
    return "post_2010"


def query_model_predictions(model, x_proc, input_df):
    n = x_proc.shape[0]

    if hasattr(model, "query"):
        return model.query(x_proc, X_raw=input_df)

    obs_model = None
    int_model = None
    n_observed_map = {}

    if hasattr(model, "tuned_observation_model"):
        obs_model = model.tuned_observation_model
    if hasattr(model, "tuned_intensity_model"):
        int_model = model.tuned_intensity_model
    if hasattr(model, "n_observed_train_"):
        n_observed_map = model.n_observed_train_

    if isinstance(model, dict):
        obs_model = obs_model or model.get("tuned_observation_model") or model.get("observation_model")
        int_model = int_model or model.get("tuned_intensity_model") or model.get("intensity_model")
        n_observed_map = n_observed_map or model.get("n_observed_train", {})

    if obs_model is not None and hasattr(obs_model, "predict_proba"):
        p_recorded = obs_model.predict_proba(x_proc)
    else:
        p_recorded = None

    if int_model is not None and hasattr(int_model, "predict_quantiles"):
        intervals = int_model.predict_quantiles(x_proc)
    elif hasattr(model, "predict_quantiles"):
        intervals = model.predict_quantiles(x_proc)
    elif hasattr(model, "predict"):
        y_pred = model.predict(x_proc)
        intervals = {}
        for i, mat in enumerate(Y_COLS):
            if getattr(y_pred, "ndim", 1) == 1:
                p50 = y_pred.astype(float)
            else:
                p50 = y_pred[:, i].astype(float)
            intervals[mat] = {"p05": p50.copy(), "p50": p50, "p95": p50.copy()}
    else:
        raise TypeError(
            f"Loaded model type {type(model).__name__} has no supported inference method (query/predict_quantiles/predict)."
        )

    results = {}
    for i, mat in enumerate(Y_COLS):
        iv = intervals.get(mat, {})
        p05 = iv.get("p05")
        p50 = iv.get("p50")
        p95 = iv.get("p95")

        if p05 is None or p50 is None or p95 is None:
            zeros = [0.0] * n
            p05 = p05 if p05 is not None else zeros
            p50 = p50 if p50 is not None else zeros
            p95 = p95 if p95 is not None else zeros

        if p_recorded is None:
            p_rec = [float("nan")] * n
            expected_reported = p50
        else:
            p_rec = p_recorded[:, i]
            expected_reported = p_rec * p50

        n_obs = int(n_observed_map.get(mat, 0)) if isinstance(n_observed_map, dict) else 0
        results[mat] = {
            "p_recorded": p_rec,
            "p05": p05,
            "p50": p50,
            "p95": p95,
            "expected_reported": expected_reported,
            "n_observed_train": n_obs,
            "coverage_warning": n_obs < 30,
            "archetype_support_level": ["unknown"] * n,
        }
    return results


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    return preprocessor, model


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

CONSTRUCTION_PERIOD_BUCKETS = [
    "pre_1945",
    "1945_1980",
    "1980_2000",
    "2000_2010",
    "post_2010",
]

st.title("Material Intensity Predictor")
st.write("Estimate material intensity percentiles (5th, 50th, and 95th) for a building.")

try:
    preprocessor, model = load_artifacts()
except Exception as exc:
    st.error(f"Error loading artifacts: {exc}")
    st.stop()

with st.sidebar:
    st.header("Building Inputs")

    construction_period = st.number_input(
        "Construction period", min_value=1900, max_value=2100, value=2015
    )
    construction_period_bucket = infer_construction_period_bucket(int(construction_period))

    st.text_input(
        "Construction period bucket",
        value=construction_period_bucket,
        disabled=True,
        help="Auto-filled from construction period year.",
    )

    typology = st.selectbox(
        "Typology",
        options=list(TYPOLOGY_MAP.keys()),
        format_func=lambda x: TYPOLOGY_MAP.get(x, x),
    )

    primary_code = st.selectbox(
        "Primary Code",
        options=list(PRIMARY_CODE_MAP.keys()),
        format_func=lambda x: PRIMARY_CODE_MAP.get(x, x),
    )

    hybrid_structure = st.selectbox(
        "Hybrid Structure",
        options=list(HYBRID_STRUCTURE_MAP.keys()),
        format_func=lambda x: HYBRID_STRUCTURE_MAP.get(x, x),
    )

    country_options = preprocessor.named_transformers_["cat"].categories_[-1]
    country = st.selectbox("Country", options=country_options)

if st.button("Predict Material Intensity", type="primary"):
    input_df = pd.DataFrame(
        [
            {
                "Construction period": construction_period,
                "Construction period bucket": construction_period_bucket,
                "Typology": typology,
                "Primary Code": primary_code,
                "Hybrid Structure": hybrid_structure,
                "Country": country,
            }
        ]
    )

    x_proc = preprocessor.transform(input_df)
    try:
        predictions = query_model_predictions(model, x_proc, input_df)
    except Exception as exc:
        st.error(f"Prediction failed for loaded model type {type(model).__name__}: {exc}")
        st.stop()

    first_mat = Y_COLS[0]
    archetype_lvl = predictions[first_mat]["archetype_support_level"][0]
    st.info(f"Archetype support level: {archetype_lvl}")

    st.subheader("Predicted Material Intensities (kg/m2)")
    cols = st.columns(len(Y_COLS))

    for col, material in zip(cols, Y_COLS):
        with col:
            p = predictions[material]
            st.markdown(f"### {material}")
            st.metric("p_recorded", f"{float(p['p_recorded'][0]):.2f}")
            p05 = float(p["p05"][0])
            p50 = float(p["p50"][0])
            p95 = float(p["p95"][0])
            render_percentile_bar(material, p05, p50, p95)
            st.metric("Expected reported (kg/m²)", f"{float(p['expected_reported'][0]):.2f}")
            n_obs = p["n_observed_train"]
            if p["coverage_warning"]:
                st.warning(f"Training records: {n_obs} — low data coverage")
            else:
                st.caption(f"Training records: {n_obs}")
else:
    st.caption("Set inputs in sidebar and click Predict.")

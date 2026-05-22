from pathlib import Path
import json
import sys
import types

import joblib
import pandas as pd
import streamlit as st

# Register model classes in sys.modules["prediction_model"] so joblib can
# deserialize model.joblib without importing the full training script
# (which would try to load the Excel dataset at import time).
import model_classes as _mc
_pm = types.ModuleType("prediction_model")
_pm.ObservationModel = _mc.ObservationModel
_pm.IntensityModel   = _mc.IntensityModel
_pm.FinalQueryModel  = _mc.FinalQueryModel
_mc.ObservationModel.__module__ = "prediction_model"
_mc.IntensityModel.__module__   = "prediction_model"
_mc.FinalQueryModel.__module__  = "prediction_model"
sys.modules.setdefault("prediction_model", _pm)

st.set_page_config(page_title="Material Intensity Predictor", layout="wide")

ARTIFACT_DIR = Path(__file__).resolve().parent

# Load material columns from model_info.json instead of importing prediction_model
with open(ARTIFACT_DIR / "model_info.json") as f:
    _model_info = json.load(f)
    Y_COLS = _model_info.get("y_cols", ["Concrete", "Glass", "Steel", "Wood", "Brick"])

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
    if year < 1980:
        return "1945_1980"
    if year < 2000:
        return "1980_2000"
    if year < 2010:
        return "2000_2010"
    return "post_2010"


def query_model_predictions(model, x_proc, input_df):
    if hasattr(model, "query"):
        return model.query(x_proc, X_raw=input_df)
    raise TypeError(
        f"Loaded model type {type(model).__module__}.{type(model).__name__} has no 'query' method. "
        "Please regenerate model.joblib from prediction_model.ipynb and redeploy."
    )


def format_model_result(predict_dict):
    """Return the raw model fields for a single-row query."""
    return {
        "archetype_n_train": int(predict_dict["archetype_n_train"][0]),
        "archetype_support_level": str(predict_dict["archetype_support_level"][0]),
        "n_observed_train": int(predict_dict["n_observed_train"]),
        "coverage_warning": bool(predict_dict["coverage_warning"]),
        "p_recorded": float(predict_dict["p_recorded"][0]),
        "p05": float(predict_dict["p05"][0]),
        "p50": float(predict_dict["p50"][0]),
        "p95": float(predict_dict["p95"][0]),
        "mean": float(predict_dict["mean"][0]),
        "expected_reported": float(predict_dict["expected_reported"][0]),
    }


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")
    model_path_candidates = [
        ARTIFACT_DIR / "model_finalquery.joblib",
        ARTIFACT_DIR / "model.joblib",
    ]
    model = None
    loaded_path = None
    for p in model_path_candidates:
        if p.exists():
            model = joblib.load(p)
            loaded_path = p
            break
    if model is None:
        raise FileNotFoundError(
            "No model artifact found. Expected one of: model_finalquery.joblib, model.joblib"
        )
    return preprocessor, model, loaded_path


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
    "C": "Concrete",
    "S": "Steel",
    "W": "Wood",
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
    preprocessor, model, model_path = load_artifacts()
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
        "Building Function",
        options=list(TYPOLOGY_MAP.keys()),
        format_func=lambda x: TYPOLOGY_MAP.get(x, x),
    )

    primary_code = st.selectbox(
        "Structural System Type",
        options=list(PRIMARY_CODE_MAP.keys()),
        format_func=lambda x: PRIMARY_CODE_MAP.get(x, x),
    )

    hybrid_structure = st.selectbox(
        "Hybrid Structure or Not",
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

    try:
        x_proc = preprocessor.transform(input_df)
    except ValueError as e:
        expected = [name for trans in preprocessor.transformers for name in (trans[2] if isinstance(trans[2], list) else [trans[2]])]
        st.error(f"Preprocessor error: {e}")
        st.write("Input columns:", list(input_df.columns))
        st.write("Preprocessor expects:", expected)
        st.stop()
    try:
        predictions = query_model_predictions(model, x_proc, input_df)
    except Exception as exc:
        st.error(f"Prediction failed for loaded model type {type(model).__name__}: {exc}")
        st.stop()

    first_mat = Y_COLS[0]
    model_result = format_model_result(predictions[first_mat])
    st.info(
        f"Archetype support level: {model_result['archetype_support_level']} "
        f"(n_train={model_result['archetype_n_train']})"
    )

    with st.expander("Raw model result", expanded=False):
        st.json(model_result)

    st.subheader("Predicted Material Intensities (kg/m²)")
    cols = st.columns(len(Y_COLS))

    for col, material in zip(cols, Y_COLS):
        with col:
            p = predictions[material]
            st.markdown(f"### {material}")
            p05 = float(p["p05"][0])
            p50 = float(p["p50"][0])
            p95 = float(p["p95"][0])
            render_percentile_bar(material, p05, p50, p95)
            st.metric("Median (p50)", f"{p50:.2f}")
            st.markdown(
                f"""
                <div style="margin-top: 0.25rem; line-height: 1.1;">
                    <div style="font-size: 0.72rem; color: #6b7280;">Database Reporting Probability</div>
                    <div style="font-size: 0.88rem; font-weight: 600; color: #111827;">{float(p['p_recorded'][0]):.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.caption("Set inputs in sidebar and click Predict.")

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
    if hasattr(model, "query"):
        return model.query(x_proc, X_raw=input_df)
    raise TypeError(
        f"Loaded model type {type(model).__module__}.{type(model).__name__} has no 'query' method. "
        "Please regenerate model.joblib from prediction_model.ipynb and redeploy."
    )


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
    preprocessor, model, model_path = load_artifacts()
except Exception as exc:
    st.error(f"Error loading artifacts: {exc}")
    st.stop()

st.caption(
    f"Loaded model: {type(model).__module__}.{type(model).__name__} from {model_path.name}"
)

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
    archetype_lvl = predictions[first_mat]["archetype_support_level"][0]
    st.info(f"Archetype support level: {archetype_lvl}")

    st.subheader("Predicted Material Intensities (kg/m2)")
    cols = st.columns(len(Y_COLS))

    for col, material in zip(cols, Y_COLS):
        with col:
            p = predictions[material]
            st.markdown(f"### {material}")
            p05 = float(p["p05"][0])
            p50 = float(p["p50"][0])
            p95 = float(p["p95"][0])
            render_percentile_bar(material, p05, p50, p95)
            mean_val = float(p["mean"][0]) if "mean" in p else p50
            st.metric("Mean (kg/m²)", f"{mean_val:.2f}")
            st.metric("Database Reporting Probability", f"{float(p['p_recorded'][0]):.2f}")
else:
    st.caption("Set inputs in sidebar and click Predict.")

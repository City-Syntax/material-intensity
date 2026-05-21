from pathlib import Path
import inspect

import joblib
import numpy as np
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

    p_recorded = None
    if hasattr(model, "predict_proba"):
        try:
            p_recorded = np.asarray(model.predict_proba(x_proc), dtype=float)
        except Exception:
            p_recorded = None

    if not hasattr(model, "predict"):
        raise TypeError(
            f"Loaded model type {type(model).__name__} has no supported inference method (query/predict)."
        )

    predict_sig = inspect.signature(model.predict)
    predict_kwargs = {}
    if "groups" in predict_sig.parameters:
        predict_kwargs["groups"] = input_df["Construction period bucket"].astype(str).tolist()

    raw_pred = model.predict(x_proc, **predict_kwargs)
    intervals = {}

    if isinstance(raw_pred, dict):
        for m, mat in enumerate(Y_COLS):
            v = raw_pred.get(mat, {})
            if isinstance(v, dict):
                p05 = np.asarray(v.get("p05", np.zeros(n)), dtype=float)
                p50 = np.asarray(v.get("p50", np.zeros(n)), dtype=float)
                p95 = np.asarray(v.get("p95", p50), dtype=float)
            else:
                arr = np.asarray(v, dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    p05, p50, p95 = arr[:, 0], arr[:, 1], arr[:, 2]
                else:
                    p50 = np.asarray(arr).reshape(-1)
                    if p50.shape[0] != n:
                        p50 = np.zeros(n)
                    p05 = p50.copy()
                    p95 = p50.copy()
            intervals[mat] = {"p05": p05, "p50": p50, "p95": p95}
    else:
        arr = np.asarray(raw_pred, dtype=float)
        if arr.ndim == 3 and arr.shape[1] == len(Y_COLS) and arr.shape[2] >= 3:
            for m, mat in enumerate(Y_COLS):
                intervals[mat] = {
                    "p05": arr[:, m, 0],
                    "p50": arr[:, m, 1],
                    "p95": arr[:, m, 2],
                }
        elif arr.ndim == 2 and arr.shape[1] == len(Y_COLS) * 3:
            arr3 = arr.reshape(n, len(Y_COLS), 3)
            for m, mat in enumerate(Y_COLS):
                intervals[mat] = {
                    "p05": arr3[:, m, 0],
                    "p50": arr3[:, m, 1],
                    "p95": arr3[:, m, 2],
                }
        elif arr.ndim == 2 and arr.shape[1] == len(Y_COLS):
            for m, mat in enumerate(Y_COLS):
                intervals[mat] = {
                    "p05": arr[:, m],
                    "p50": arr[:, m],
                    "p95": arr[:, m],
                }
        else:
            raise TypeError(
                f"Unsupported predict output shape {arr.shape} for model type {type(model).__name__}."
            )

    if p_recorded is None or p_recorded.shape != (n, len(Y_COLS)):
        p_recorded = np.ones((n, len(Y_COLS)), dtype=float)

    results = {}
    for m, mat in enumerate(Y_COLS):
        p05 = np.asarray(intervals[mat]["p05"], dtype=float)
        p50 = np.asarray(intervals[mat]["p50"], dtype=float)
        p95 = np.asarray(intervals[mat]["p95"], dtype=float)
        p_rec = p_recorded[:, m]
        results[mat] = {
            "p_recorded": p_rec,
            "p05": p05,
            "p50": p50,
            "p95": p95,
            "expected_reported": p_rec * p50,
            "n_observed_train": 0,
            "coverage_warning": False,
            "archetype_support_level": np.array(["unknown"] * n, dtype=object),
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

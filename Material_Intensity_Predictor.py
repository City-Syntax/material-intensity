from pathlib import Path
import json
import sys
import types
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Material Intensity Predictor", layout="wide")

ARTIFACT_DIR = Path(__file__).resolve().parent
DEFAULT_Y_COLS = ["Concrete", "Glass", "Steel", "Wood", "Brick"]
DEFAULT_LOW_OBS_THRESHOLD = 30


def _register_pickle_compat_classes():
    """Register notebook class shims expected by joblib pickles.

    Models trained in Jupyter are often serialized with module names like
    `main` / `__main__`. Streamlit Cloud may load them in a different module
    context, so we expose compatible class names before joblib.load().
    """

    class ObservationModel:
        def predict_proba(self, X):
            y_cols = getattr(self, "y_cols", None) or list(getattr(self, "models_", {}).keys()) or DEFAULT_Y_COLS
            n = X.shape[0]
            proba = np.zeros((n, len(y_cols)), dtype=np.float64)
            models = getattr(self, "models_", {}) or {}
            trivial = getattr(self, "trivial_proba_", {}) or {}

            for m, material in enumerate(y_cols):
                clf = models.get(material)
                if clf is None:
                    proba[:, m] = float(trivial.get(material, 0.0))
                else:
                    proba[:, m] = clf.predict_proba(X)[:, 1]
            return proba

    class IntensityModel:
        def predict_quantiles(self, X):
            y_cols = getattr(self, "y_cols", None) or list(getattr(self, "models_", {}).keys()) or DEFAULT_Y_COLS
            n = X.shape[0]
            result = {}
            models = getattr(self, "models_", {}) or {}

            for material in y_cols:
                mdl = models.get(material)
                if mdl is None:
                    result[material] = {
                        "p05": np.zeros(n),
                        "p50": np.zeros(n),
                        "p95": np.zeros(n),
                    }
                    continue

                pq = mdl.predict(X)
                p05 = np.maximum(np.expm1(pq[:, 0]), 0.0)
                p50 = np.maximum(np.expm1(pq[:, 1]), 0.0)
                p95 = np.maximum(np.expm1(pq[:, 2]), 0.0)
                result[material] = {
                    "p05": p05,
                    "p50": p50,
                    "p95": p95,
                }
            return result

    class FinalQueryModel:
        def query(self, X_proc):
            obs_model = getattr(self, "tuned_observation_model", None)
            int_model = getattr(self, "tuned_intensity_model", None)
            if obs_model is None or int_model is None:
                raise AttributeError("Missing tuned_observation_model or tuned_intensity_model")

            p_recorded = obs_model.predict_proba(X_proc)
            intervals = int_model.predict_quantiles(X_proc)

            y_cols = getattr(self, "y_cols", None) or getattr(self, "y_cols_", None) or DEFAULT_Y_COLS
            low_obs = getattr(self, "LOW_OBS_THRESHOLD", None)
            if low_obs is None:
                low_obs = getattr(self, "low_obs_threshold", DEFAULT_LOW_OBS_THRESHOLD)
            n_support_map = getattr(self, "n_support_", None) or getattr(self, "n_observed_train_", {}) or {}

            result = {}
            for m, mat in enumerate(y_cols):
                n_sup = int(n_support_map.get(mat, 0))
                p_rec = p_recorded[:, m]
                sup_score = min(1.0, n_sup / max(float(low_obs), 1.0))

                mat_iv = intervals.get(mat, {})
                p05 = np.asarray(mat_iv.get("p05", np.zeros(X_proc.shape[0])))
                p50 = np.asarray(mat_iv.get("p50", np.zeros(X_proc.shape[0])))
                p95 = np.asarray(mat_iv.get("p95", np.zeros(X_proc.shape[0])))

                result[mat] = {
                    "p_recorded": p_rec,
                    "probability_unrecorded": 1.0 - p_rec,
                    "conditional_p05": p05,
                    "conditional_p50": p50,
                    "conditional_p95": p95,
                    "recording_adjusted_median": p_rec * p50,
                    "n_support": n_sup,
                    "support_score": sup_score,
                    "support_warning": n_sup < low_obs,
                    "query_confidence": p_rec * sup_score,
                }
            return result

    for mod_name in ("main", "__main__"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            mod = types.ModuleType(mod_name)
            sys.modules[mod_name] = mod

        # Always overwrite to avoid stale class definitions surviving across reruns.
        setattr(mod, "ObservationModel", ObservationModel)
        setattr(mod, "IntensityModel", IntensityModel)
        setattr(mod, "FinalQueryModel", FinalQueryModel)


def _load_material_columns(model):
    model_info_path = ARTIFACT_DIR / "model_info.json"
    if model_info_path.exists():
        try:
            with model_info_path.open("r", encoding="utf-8") as f:
                info = json.load(f)
            y_cols = info.get("y_cols")
            if isinstance(y_cols, list) and y_cols:
                return [str(c) for c in y_cols]
        except Exception:
            pass

    for attr in ("y_cols", "Y_COLS"):
        if hasattr(model, attr):
            cols = getattr(model, attr)
            if isinstance(cols, (list, tuple)) and cols:
                return [str(c) for c in cols]

    return DEFAULT_Y_COLS


# Register shim classes at module load time so they are always present before
# joblib.load() runs, even when @st.cache_resource skips the function body.
_register_pickle_compat_classes()


@st.cache_resource
def load_artifacts():
    _register_pickle_compat_classes()  # re-register in case cache was cleared
    preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    y_cols = _load_material_columns(model)
    return preprocessor, model, y_cols


def _as_1d_array(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


def _normalize_predictions(raw_predictions, n_rows, y_cols):
    """Normalize model outputs to a common schema used by the app UI."""
    if isinstance(raw_predictions, dict):
        normalized = {}
        for material in y_cols:
            mat_pred = raw_predictions.get(material, {})

            p05 = _as_1d_array(
                mat_pred.get("conditional_p05", mat_pred.get("p05", mat_pred.get("p5", np.zeros(n_rows))))
            )
            p50 = _as_1d_array(
                mat_pred.get("conditional_p50", mat_pred.get("p50", np.zeros(n_rows)))
            )
            p95 = _as_1d_array(
                mat_pred.get("conditional_p95", mat_pred.get("p95", np.zeros(n_rows)))
            )
            p_rec = _as_1d_array(
                mat_pred.get("p_recorded", mat_pred.get("p_presence", np.ones(n_rows)))
            )

            support_score_present = "support_score" in mat_pred
            n_support_present = "n_support" in mat_pred or "n_observed_train" in mat_pred
            support_warning_present = "support_warning" in mat_pred or "coverage_warning" in mat_pred

            n_support = mat_pred.get("n_support", mat_pred.get("n_observed_train", 0))
            n_support = int(np.asarray(n_support).reshape(-1)[0])

            support_score = mat_pred.get("support_score", 0.0)
            support_score = float(np.asarray(support_score).reshape(-1)[0])

            support_warning = mat_pred.get("support_warning", mat_pred.get("coverage_warning", False))
            support_warning = bool(np.asarray(support_warning).reshape(-1)[0])

            # If diagnostics are absent (e.g., older model API), avoid misleading
            # "100% support" defaults.
            if not support_score_present and not n_support_present:
                support_score = 0.0
                support_warning = True
            elif n_support <= 0 and not support_score_present:
                support_score = 0.0
                if not support_warning_present:
                    support_warning = True

            normalized[material] = {
                "conditional_p05": p05,
                "conditional_p50": p50,
                "conditional_p95": p95,
                "p_recorded": p_rec,
                "support_score": support_score,
                "n_support": n_support,
                "support_warning": support_warning,
            }
        return normalized

    # Fallback for array-like predictions (e.g., plain regressors returning medians).
    pred_arr = np.asarray(raw_predictions)
    if pred_arr.ndim == 1:
        pred_arr = pred_arr.reshape(1, -1)

    if pred_arr.shape[1] != len(y_cols):
        raise ValueError(
            f"Unexpected prediction shape {pred_arr.shape}; expected second dimension {len(y_cols)}"
        )

    normalized = {}
    for idx, material in enumerate(y_cols):
        median = np.maximum(pred_arr[:, idx], 0.0)
        normalized[material] = {
            "conditional_p05": median,
            "conditional_p50": median,
            "conditional_p95": median,
            "p_recorded": np.ones(n_rows),
            "support_score": 0.0,
            "n_support": 0,
            "support_warning": True,
        }
    return normalized


def run_inference(model, x_proc, input_df, y_cols):
    """Run inference across model versions exposing query() or predict()."""
    if hasattr(model, "query"):
        raw_predictions = model.query(x_proc)
    elif hasattr(model, "predict"):
        groups = input_df["Primary Code"].astype(str).to_numpy()
        try:
            raw_predictions = model.predict(x_proc, groups, alpha=0.10)
        except TypeError:
            raw_predictions = model.predict(x_proc)
    else:
        raise AttributeError("Loaded model exposes neither query() nor predict().")

    return _normalize_predictions(raw_predictions, n_rows=x_proc.shape[0], y_cols=y_cols)


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

st.title("Material Intensity Predictor")
st.write(
    "Estimate conditional material intensity (MI) percentiles (5th, 50th, and 95th) for a building, "
    "together with database availability and data-support diagnostics."
)

try:
    preprocessor, model, y_cols = load_artifacts()
except Exception as exc:
    st.error(f"Error loading artifacts: {exc}")
    st.stop()

with st.sidebar:
    st.header("Building Features (Archetype Definition)")

    construction_period = st.number_input(
        "Construction Year", min_value=1900, max_value=2100, value=2015
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

    country_options = preprocessor.named_transformers_["cat"].categories_[3]
    country = st.selectbox("Country", options=country_options)

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

    x_proc = preprocessor.transform(input_df)
    try:
        predictions = run_inference(model, x_proc, input_df, y_cols)
    except Exception as exc:
        st.error(f"Error during prediction: {exc}")
        st.stop()

    st.subheader("Predicted Material Intensities (kg/m²)")
    cols = st.columns(len(y_cols))

    for col, material in zip(cols, y_cols):
        with col:
            p = predictions[material]
            st.markdown(f"### {material}")

            st.metric("5th percentile (kg/m²)", f"{float(p['conditional_p05'][0]):.2f}")
            st.metric("Median (kg/m²)",          f"{float(p['conditional_p50'][0]):.2f}")
            st.metric("95th percentile (kg/m²)", f"{float(p['conditional_p95'][0]):.2f}")

            st.divider()

            p_rec = float(p["p_recorded"][0])
            st.metric(
                "Database availability",
                f"{p_rec:.0%}",
                help="Presence probability that this material MI is recorded for these archetypes.",
            )

    # ── Building-level data support diagnostic ────────────────────────
    sup_scores = {mat: float(predictions[mat]["support_score"])  for mat in y_cols}
    n_supports = {mat: int(predictions[mat]["n_support"])        for mat in y_cols}
    any_warn   = any(predictions[mat]["support_warning"]         for mat in y_cols)
    min_score  = min(sup_scores.values())
    unique_n_support = sorted(set(n_supports.values()))
    archetype_n_support = unique_n_support[0] if len(unique_n_support) == 1 else None

    with st.expander("Data support for this building archetype", expanded=True):
        dc1, dc2 = st.columns([1, 3])
        with dc1:
            st.metric(
                "Data support",
                f"{min_score:.0%}",
                help=(
                    "How much similar training data supports this query. "
                    "Shown as the most conservative (minimum) score across all materials."
                ),
            )
            if archetype_n_support is not None:
                st.metric(
                    "Archetype support count",
                    f"{archetype_n_support}",
                    help="Number of training buildings supporting this archetype.",
                )
            if any_warn:
                st.warning("Support warning: sparse or weakly supported archetype.", icon="⚠️")
        with dc2:
            if archetype_n_support is None:
                st.caption("Training observations by material (counts differ):")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Material":      list(n_supports.keys()),
                            "n_support":     list(n_supports.values()),
                            "support_score": [f"{v:.0%}" for v in sup_scores.values()],
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("`n_support` is archetype-level for this query (same across materials).")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Material": list(sup_scores.keys()),
                            "support_score": [f"{v:.0%}" for v in sup_scores.values()],
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

else:
    st.caption("Set inputs in sidebar and click Predict.")

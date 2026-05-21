"""prediction_model.py â€” script version of prediction_model.ipynb.

Runs the full pipeline: data preparation, model definition, Optuna tuning,
validation-set conformal calibration, evaluation, visualisation, and artifact export.
"""

import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    median_absolute_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

SEED = 42
MIN_OBSERVED_TARGETS = 2
LOW_OBS_THRESHOLD = 30
CAL_ALPHA = 0.10
IPS_MIN_PROBA = 0.05
IPS_MAX_WEIGHT = 20.0
ARCHETYPE_HIGH_SUPPORT = 30
ARCHETYPE_MEDIUM_SUPPORT = 10
ARCHETYPE_LOW_SUPPORT = 3

N_TRIALS = 40  # Optuna trials per stage

PERIOD_BUCKETS = ["pre_1945", "1945_1980", "1980_2000", "2000_2010", "post_2010"]
X_cols = [
    "Construction period",
    "Construction period bucket",
    "Typology",
    "Primary Code",
    "Hybrid Structure",
    "Country",
]
y_cols = ["Concrete", "Glass", "Steel", "Wood", "Brick"]


def to_period_bucket(year_series):
    year = pd.to_numeric(year_series, errors="coerce")
    return pd.cut(
        year,
        bins=[-np.inf, 1945, 1980, 2000, 2010, np.inf],
        labels=PERIOD_BUCKETS,
        right=False,
    )


def archetype_support_level(n_rows):
    if n_rows >= ARCHETYPE_HIGH_SUPPORT:
        return "high"
    if n_rows >= ARCHETYPE_MEDIUM_SUPPORT:
        return "medium"
    if n_rows >= ARCHETYPE_LOW_SUPPORT:
        return "low"
    if n_rows >= 1:
        return "very_low"
    return "none"

# â”€â”€ Visualisation style â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_C1   = "#4C72B0"
_C2   = "#DD8452"
_REF  = "#C44E52"
_GRAY = "#AAAAAA"

plt.rcParams.update({
    "figure.dpi":      110,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         10,
})


def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


set_global_seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==========================================================
# Data Preparation
# ==========================================================

def prepare_data(
    file_path="Integrated_MI_database_add_Singapore.xlsx",
    test_size=0.30,
    val_size=0.50,
    min_observed_targets=MIN_OBSERVED_TARGETS,
    random_state=SEED,
):
    file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path

    df = pd.read_excel(file_path)
    df["Construction period"] = pd.to_numeric(df["Construction period"], errors="coerce")
    df["Construction period bucket"] = to_period_bucket(df["Construction period"])
    df = df.dropna(subset=X_cols).reset_index(drop=True)

    target_mask_df = df[y_cols].notna()
    df = df.loc[target_mask_df.sum(axis=1) >= min_observed_targets].reset_index(drop=True)

    X = df[X_cols].copy()
    y_raw_df = df[y_cols].copy()
    y_mask = y_raw_df.notna().to_numpy(dtype=bool)

    X_train, X_temp, y_train_df, y_temp_df, y_train_mask, y_temp_mask = train_test_split(
        X, y_raw_df, y_mask, test_size=test_size, random_state=random_state
    )
    X_val, X_test, y_val_df, y_test_df, y_val_mask, y_test_mask = train_test_split(
        X_temp, y_temp_df, y_temp_mask, test_size=val_size, random_state=random_state
    )

    for df_ in [X_train, X_val, X_test, y_train_df, y_val_df, y_test_df]:
        df_.reset_index(drop=True, inplace=True)

    y_train_raw = y_train_df.to_numpy(dtype=np.float64).copy()
    y_val_raw   = y_val_df.to_numpy(dtype=np.float64).copy()
    y_test_raw  = y_test_df.to_numpy(dtype=np.float64).copy()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), ["Construction period"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["Construction period bucket", "Typology", "Primary Code", "Hybrid Structure", "Country"]),
    ])
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc   = preprocessor.transform(X_val)
    X_test_proc  = preprocessor.transform(X_test)

    return dict(
        X_train_proc=X_train_proc, X_val_proc=X_val_proc, X_test_proc=X_test_proc,
        X_train_raw=X_train,       X_val_raw=X_val,        X_test_raw=X_test,
        y_train_raw=y_train_raw,   y_val_raw=y_val_raw,    y_test_raw=y_test_raw,
        y_train_df=y_train_df,     y_val_df=y_val_df,      y_test_df=y_test_df,
        y_train_mask=y_train_mask, y_val_mask=y_val_mask,  y_test_mask=y_test_mask,
        preprocessor=preprocessor,
        kept_rows=len(df),
        min_observed_targets=min_observed_targets,
    )


# ==========================================================
# Stage 1 â€” Observation Model
# ==========================================================

class ObservationModel:
    """Per-material XGBoost classifier predicting P(recorded | x)."""

    def __init__(
        self,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=SEED,
    ):
        self.xgb_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            objective="binary:logistic",
            verbosity=0,
        )
        self.models_ = {}
        self.trivial_proba_ = {}

    def fit(self, X, y_observed):
        self.models_ = {}
        self.trivial_proba_ = {}
        for m, material in enumerate(y_cols):
            s = y_observed[:, m].astype(int)
            p = float(s.mean())
            if p == 0.0 or p == 1.0:
                self.trivial_proba_[material] = p
                self.models_[material] = None
            else:
                base = XGBClassifier(**self.xgb_params)
                clf  = CalibratedClassifierCV(base, method="sigmoid", cv=5)
                clf.fit(X, s)
                self.models_[material] = clf
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        proba = np.zeros((n, len(y_cols)), dtype=np.float64)
        for m, material in enumerate(y_cols):
            clf = self.models_.get(material)
            if clf is None:
                proba[:, m] = self.trivial_proba_.get(material, 0.0)
            else:
                proba[:, m] = clf.predict_proba(X)[:, 1]
        return proba


# ==========================================================
# Stage 2 â€” Intensity Model
# ==========================================================

class IntensityModel:
    """Per-material quantile XGBoost estimating conditional intensity (kg/mÂ²)."""

    ALPHAS = [0.05, 0.50, 0.95]

    def __init__(self, n_estimators=300, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.0, reg_lambda=1.0, random_state=SEED):
        self.xgb_params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
            reg_lambda=reg_lambda, random_state=random_state,
            verbosity=0,
        )
        self.models_ = {}

    @staticmethod
    def _inverse_propensity_weights(p_recorded_obs):
        p = np.clip(np.asarray(p_recorded_obs, dtype=np.float64), IPS_MIN_PROBA, 1.0)
        w = 1.0 / p
        w = np.minimum(w, IPS_MAX_WEIGHT)
        # Keep mean weight near 1.0 for stable optimization across materials.
        return w / np.mean(w)

    def fit(self, X, y_raw, y_observed, p_recorded=None):
        self.models_ = {}
        for m, material in enumerate(y_cols):
            obs = y_observed[:, m]
            if obs.sum() < 2:
                self.models_[material] = None
                continue
            y_log = np.log1p(y_raw[obs, m])
            if p_recorded is None:
                sw = np.ones(obs.sum(), dtype=np.float64)
            else:
                sw = self._inverse_propensity_weights(p_recorded[obs, m])
            mdl = XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=self.ALPHAS,
                **self.xgb_params,
            )
            mdl.fit(X[obs], y_log, sample_weight=sw)
            self.models_[material] = mdl
        return self

    def predict_quantiles(self, X):
        n = X.shape[0]
        result = {}
        for material in y_cols:
            mdl = self.models_.get(material)
            if mdl is None:
                result[material] = {"p05": np.zeros(n), "p50": np.zeros(n), "p95": np.zeros(n)}
                continue
            pq   = mdl.predict(X)
            p_lo = np.maximum(np.expm1(pq[:, 0]), 0.0)
            p50  = np.maximum(np.expm1(pq[:, 1]), 0.0)
            p_hi = np.maximum(np.expm1(pq[:, 2]), 0.0)
            result[material] = {"p05": p_lo, "p50": p50, "p95": p_hi}
        return result


# ==========================================================
# FinalQueryModel â€” two-stage wrapper
# ==========================================================

class FinalQueryModel:
    """Two-stage database-informed query model.

    Stage 1 (tuned_observation_model): P(recorded | x) per material.
    Stage 2 (tuned_intensity_model):   conditional intensity quantiles (kg/mÂ²).

    query() output per material
    ---------------------------
    p_recorded        Stage 1 P(intensity is recorded in the database)
    support_confidence alias of p_recorded for confidence/data-support display
    p05, p50, p95     conditional intensity quantiles (kg/mÂ²)
    expected_reported p_recorded * p50
    n_observed_train  training rows with this material recorded
    coverage_warning  True if n_observed_train < LOW_OBS_THRESHOLD
    archetype_n_train training rows matching the full input archetype (all X_cols)
    archetype_support_level qualitative support label from archetype_n_train
    """

    def __init__(self, observation_params=None, intensity_params=None):
        _default = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.0, reg_lambda=1.0, random_state=SEED)
        self.tuned_observation_model = ObservationModel(**(observation_params or _default))
        self.tuned_intensity_model   = IntensityModel(**(intensity_params   or _default))

    @staticmethod
    def _archetype_key_row(row):
        return tuple("<NA>" if pd.isna(v) else str(v) for v in row)

    def _build_archetype_count_map(self, X_raw):
        if X_raw is None:
            return {}
        xdf = pd.DataFrame(X_raw).copy()
        xdf = xdf[X_cols].copy()
        keys = xdf.apply(self._archetype_key_row, axis=1)
        counts = keys.value_counts()
        return {k: int(v) for k, v in counts.items()}

    def fit(self, X_proc, y_raw, y_mask, X_raw=None):
        y_observed = y_mask
        self.n_observed_train_ = {
            mat: int(y_observed[:, m].sum()) for m, mat in enumerate(y_cols)
        }
        self.archetype_count_map_ = self._build_archetype_count_map(X_raw)
        self.tuned_observation_model.fit(X_proc, y_observed)
        p_recorded = self.tuned_observation_model.predict_proba(X_proc)
        self.tuned_intensity_model.fit(X_proc, y_raw, y_observed, p_recorded=p_recorded)
        return self

    def query(self, X_proc, X_raw=None):
        p_recorded = self.tuned_observation_model.predict_proba(X_proc)
        intervals  = self.tuned_intensity_model.predict_quantiles(X_proc)
        n = X_proc.shape[0]
        if X_raw is None or not getattr(self, "archetype_count_map_", None):
            archetype_n = np.full(n, np.nan)
            archetype_lvl = np.array(["unknown"] * n, dtype=object)
        else:
            xdf = pd.DataFrame(X_raw).copy()
            xdf = xdf[X_cols].copy()
            keys = xdf.apply(self._archetype_key_row, axis=1)
            archetype_n = np.array(
                [self.archetype_count_map_.get(k, 0) for k in keys],
                dtype=np.int64,
            )
            archetype_lvl = np.array([archetype_support_level(int(v)) for v in archetype_n], dtype=object)
        result = {}
        for m, mat in enumerate(y_cols):
            n_obs = self.n_observed_train_.get(mat, 0)
            result[mat] = {
                "p_recorded":        p_recorded[:, m],
                "support_confidence": p_recorded[:, m],
                "p05":               intervals[mat]["p05"],
                "p50":               intervals[mat]["p50"],
                "p95":               intervals[mat]["p95"],
                "expected_reported": p_recorded[:, m] * intervals[mat]["p50"],
                "n_observed_train":  n_obs,
                "coverage_warning":  n_obs < LOW_OBS_THRESHOLD,
                "archetype_n_train": archetype_n,
                "archetype_support_level": archetype_lvl,
            }
        return result


# ==========================================================
# Conformal calibration helper
# ==========================================================

def _conformal_offset(y_true, p05, p95, alpha=CAL_ALPHA):
    """Split conformal offset ensuring >= 1-alpha coverage."""
    scores = np.maximum(p05 - y_true, y_true - p95)
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 0.0
    k = int(np.ceil((scores.size + 1) * (1.0 - alpha)))
    k = max(1, min(k, scores.size))
    return float(np.sort(scores)[k - 1])


# ==========================================================
# Main workflow
# ==========================================================

if __name__ == "__main__":

    # â”€â”€ Data preparation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    set_global_seed(SEED)
    data = prepare_data()

    print("Data preparation complete.")
    print(f"X_train: {data['X_train_proc'].shape}  y_train: {data['y_train_raw'].shape}")
    print(f"X_val:   {data['X_val_proc'].shape}  y_val:   {data['y_val_raw'].shape}")
    print(f"X_test:  {data['X_test_proc'].shape}  y_test:  {data['y_test_raw'].shape}")
    print(f"Rows kept: {data['kept_rows']}  (min observed targets: {data['min_observed_targets']})")

    # â”€â”€ Hyperparameter tuning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _obs_objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int(  "n_estimators",     100, 500),
            max_depth        = trial.suggest_int(  "max_depth",          3,   6),
            learning_rate    = trial.suggest_float("learning_rate",   0.01, 0.2, log=True),
            subsample        = trial.suggest_float("subsample",        0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha",        0.0, 2.0),
            reg_lambda       = trial.suggest_float("reg_lambda",       0.5, 5.0),
            random_state     = SEED,
        )
        m = ObservationModel(**params)
        m.fit(data["X_train_proc"], data["y_train_mask"])
        p = m.predict_proba(data["X_val_proc"])
        aucs = [
            roc_auc_score(data["y_val_mask"][:, i].astype(int), p[:, i])
            for i in range(len(y_cols))
            if 0 < data["y_val_mask"][:, i].sum() < len(data["y_val_mask"])
        ]
        return -np.mean(aucs) if aucs else 0.0

    obs_study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
    obs_study.optimize(_obs_objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_observation_params = {**obs_study.best_params, "random_state": SEED}
    print(f"Stage 1 best  mean AUC = {-obs_study.best_value:.4f}")

    obs_for_ips = ObservationModel(**best_observation_params)
    obs_for_ips.fit(data["X_train_proc"], data["y_train_mask"])
    p_rec_train_for_ips = obs_for_ips.predict_proba(data["X_train_proc"])

    def _int_objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int(  "n_estimators",     100, 500),
            max_depth        = trial.suggest_int(  "max_depth",          3,   6),
            learning_rate    = trial.suggest_float("learning_rate",   0.01, 0.2, log=True),
            subsample        = trial.suggest_float("subsample",        0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha",        0.0, 2.0),
            reg_lambda       = trial.suggest_float("reg_lambda",       0.5, 5.0),
            random_state     = SEED,
        )
        m = IntensityModel(**params)
        m.fit(
            data["X_train_proc"],
            data["y_train_raw"],
            data["y_train_mask"],
            p_recorded=p_rec_train_for_ips,
        )
        ivs = m.predict_quantiles(data["X_val_proc"])
        maes = [
            mean_absolute_error(data["y_val_raw"][data["y_val_mask"][:, i], i],
                                ivs[mat]["p50"][data["y_val_mask"][:, i]])
            for i, mat in enumerate(y_cols)
            if data["y_val_mask"][:, i].sum() >= 2
        ]
        return np.mean(maes) if maes else 0.0

    int_study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
    int_study.optimize(_int_objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_intensity_params = {**int_study.best_params, "random_state": SEED}
    print(f"Stage 2 best  mean MAE = {int_study.best_value:.4f}")

    # â”€â”€ Train final model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    set_global_seed(SEED)
    final_query_model = FinalQueryModel(
        observation_params=best_observation_params,
        intensity_params=best_intensity_params,
    )
    final_query_model.fit(
        data["X_train_proc"],
        data["y_train_raw"],
        data["y_train_mask"],
        X_raw=data["X_train_raw"],
    )
    tuned_observation_model = final_query_model.tuned_observation_model
    tuned_intensity_model   = final_query_model.tuned_intensity_model

    print("Final model trained.")
    for mat, n in final_query_model.n_observed_train_.items():
        warn = "  *** low-data warning" if n < LOW_OBS_THRESHOLD else ""
        print(f"  {mat:<10}  {n:>4} training observations{warn}")

    # â”€â”€ Stage 1 evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p_rec_test = tuned_observation_model.predict_proba(data["X_test_proc"])
    y_test_obs = data["y_test_mask"]

    print("\nStage 1 â€” Recording probability model  (test set)")
    print(f"{'Material':<12}  {'AUC':>6}  {'Accuracy':>9}  {'Precision':>10}  "
          f"{'Recall':>7}  {'F1':>6}  {'n_pos':>6}  {'n_neg':>6}")
    print("-" * 72)
    for m, mat in enumerate(y_cols):
        y_true = y_test_obs[:, m].astype(int)
        y_prob = p_rec_test[:, m]
        y_pred = (y_prob > 0.5).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            print(f"{mat:<12}  {'â€”':>6}")
            continue
        print(f"{mat:<12}  "
              f"{roc_auc_score(y_true, y_prob):>6.3f}  "
              f"{accuracy_score(y_true, y_pred):>9.3f}  "
              f"{precision_score(y_true, y_pred, zero_division=0):>10.3f}  "
              f"{recall_score(y_true, y_pred):>7.3f}  "
              f"{f1_score(y_true, y_pred, zero_division=0):>6.3f}  "
              f"{int(y_true.sum()):>6}  {int((1 - y_true).sum()):>6}")

    # â”€â”€ Stage 2 point estimates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    intervals_test = tuned_intensity_model.predict_quantiles(data["X_test_proc"])

    print("\nStage 2 â€” Conditional intensity model  (test set, observed rows only)")
    print(f"{'Material':<12}  {'n_obs':>6}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}  {'MedianAE':>9}")
    print("-" * 60)
    for m, mat in enumerate(y_cols):
        obs = data["y_test_mask"][:, m]
        if obs.sum() < 2:
            print(f"{mat:<12}  {'â€”':>6}")
            continue
        y_true = data["y_test_raw"][obs, m]
        y_hat  = intervals_test[mat]["p50"][obs]
        print(f"{mat:<12}  {int(obs.sum()):>6}  "
              f"{mean_absolute_error(y_true, y_hat):>8.2f}  "
              f"{np.sqrt(mean_squared_error(y_true, y_hat)):>8.2f}  "
              f"{r2_score(y_true, y_hat):>8.3f}  "
              f"{median_absolute_error(y_true, y_hat):>9.2f}")

    # â”€â”€ Validation-set conformal calibration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    intervals_val = tuned_intensity_model.predict_quantiles(data["X_val_proc"])
    validation_offsets = {}
    for m, mat in enumerate(y_cols):
        obs = data["y_val_mask"][:, m]
        if obs.sum() < 2:
            validation_offsets[mat] = 0.0
            continue
        y_true_v = data["y_val_raw"][obs, m]
        p05_v    = intervals_val[mat]["p05"][obs]
        p95_v    = intervals_val[mat]["p95"][obs]
        validation_offsets[mat] = _conformal_offset(y_true_v, p05_v, p95_v)

    intervals_test_calibrated = {
        mat: {
            "p05": intervals_test[mat]["p05"] - validation_offsets[mat],
            "p50": intervals_test[mat]["p50"],
            "p95": intervals_test[mat]["p95"] + validation_offsets[mat],
        }
        for mat in y_cols
    }

    print("\nValidation-set conformal offsets (estimated on validation only)")
    print(f"{'Material':<12}  {'n_val':>6}  {'Offset':>10}")
    print("-" * 32)
    for m, mat in enumerate(y_cols):
        n_val_obs = int(data["y_val_mask"][:, m].sum())
        print(f"{mat:<12}  {n_val_obs:>6}  {validation_offsets[mat]:>10.2f}")

    print("\nStage 2 â€” 90% prediction interval  (test set, observed rows only)")
    print(f"{'Material':<12}  {'n_obs':>6}  {'CovU':>7}  {'CovC':>7}  "
          f"{'MeanWU':>10}  {'MeanWC':>10}  {'MedWU':>10}  {'MedWC':>10}")
    print("-" * 82)
    for m, mat in enumerate(y_cols):
        obs = data["y_test_mask"][:, m]
        if obs.sum() < 2:
            print(f"{mat:<12}  {'â€”':>6}")
            continue
        y_true = data["y_test_raw"][obs, m]
        p05_u  = intervals_test[mat]["p05"][obs]
        p95_u  = intervals_test[mat]["p95"][obs]
        p05_c  = intervals_test_calibrated[mat]["p05"][obs]
        p95_c  = intervals_test_calibrated[mat]["p95"][obs]
        cov_u  = float(((y_true >= p05_u) & (y_true <= p95_u)).mean())
        cov_c  = float(((y_true >= p05_c) & (y_true <= p95_c)).mean())
        w_u    = p95_u - p05_u
        w_c    = p95_c - p05_c
        print(f"{mat:<12}  {int(obs.sum()):>6}  {cov_u:>7.3f}  {cov_c:>7.3f}  "
              f"{w_u.mean():>10.2f}  {w_c.mean():>10.2f}  "
              f"{np.median(w_u):>10.2f}  {np.median(w_c):>10.2f}")

    # â”€â”€ Baseline comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\nBaseline comparison  (test set, observed rows only)")
    print(f"{'Material':<12}  {'Stage2 MAE':>12}  {'Median MAE':>12}  "
          f"{'Ridge MAE':>10}  {'RF MAE':>8}")
    print("-" * 62)
    for m, mat in enumerate(y_cols):
        obs_tr = data["y_train_mask"][:, m]
        obs_te = data["y_test_mask"][:, m]
        if obs_tr.sum() < 2 or obs_te.sum() < 2:
            print(f"{mat:<12}  {'â€”':>12}")
            continue
        X_tr = data["X_train_proc"][obs_tr]
        y_tr = data["y_train_raw"][obs_tr, m]
        X_te = data["X_test_proc"][obs_te]
        y_te = data["y_test_raw"][obs_te, m]

        mae_s2  = mean_absolute_error(y_te, intervals_test[mat]["p50"][obs_te])
        mae_med = mean_absolute_error(y_te, np.full(obs_te.sum(), np.median(y_tr)))

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, np.log1p(y_tr))
        mae_ridge = mean_absolute_error(y_te, np.maximum(np.expm1(ridge.predict(X_te)), 0.0))

        rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        mae_rf = mean_absolute_error(y_te, rf.predict(X_te))

        print(f"{mat:<12}  {mae_s2:>12.2f}  {mae_med:>12.2f}  "
              f"{mae_ridge:>10.2f}  {mae_rf:>8.2f}")

    # â”€â”€ Final summary table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\nFinal Summary â€” Test Set")
    hdr = (f"{'Material':<12}  {'n_obs_train':>12}  {'AUC':>6}  "
           f"{'MAE':>8}  {'RMSE':>8}  {'R2':>6}  "
           f"{'CovU':>7}  {'CovC':>7}  {'MeanWU':>10}  {'MeanWC':>10}  "
           f"{'MedWU':>10}  {'MedWC':>10}")
    print(hdr)
    print("-" * len(hdr))

    summary_rows = []
    for m, mat in enumerate(y_cols):
        n_obs_train = final_query_model.n_observed_train_.get(mat, 0)

        y_cls = data["y_test_mask"][:, m].astype(int)
        y_prb = p_rec_test[:, m]
        auc = (round(roc_auc_score(y_cls, y_prb), 3)
               if 0 < y_cls.sum() < len(y_cls) else float("nan"))

        obs_te = data["y_test_mask"][:, m]
        if obs_te.sum() < 2:
            mae = rmse = r2 = cov_u = cov_c = mean_w_u = mean_w_c = med_w_u = med_w_c = float("nan")
        else:
            y_te   = data["y_test_raw"][obs_te, m]
            p50_te = intervals_test[mat]["p50"][obs_te]
            p05_u  = intervals_test[mat]["p05"][obs_te]
            p95_u  = intervals_test[mat]["p95"][obs_te]
            p05_c  = intervals_test_calibrated[mat]["p05"][obs_te]
            p95_c  = intervals_test_calibrated[mat]["p95"][obs_te]
            mae    = round(mean_absolute_error(y_te, p50_te), 2)
            rmse   = round(np.sqrt(mean_squared_error(y_te, p50_te)), 2)
            r2     = round(r2_score(y_te, p50_te), 3)
            cov_u  = round(float(((y_te >= p05_u) & (y_te <= p95_u)).mean()), 3)
            cov_c  = round(float(((y_te >= p05_c) & (y_te <= p95_c)).mean()), 3)
            w_u    = p95_u - p05_u
            w_c    = p95_c - p05_c
            mean_w_u = round(float(w_u.mean()), 2)
            mean_w_c = round(float(w_c.mean()), 2)
            med_w_u  = round(float(np.median(w_u)), 2)
            med_w_c  = round(float(np.median(w_c)), 2)

        summary_rows.append({"Material": mat, "n_obs_train": n_obs_train, "AUC": auc,
                              "MAE": mae, "RMSE": rmse, "R2": r2,
                              "CovU": cov_u, "CovC": cov_c,
                              "MeanWU": mean_w_u, "MeanWC": mean_w_c,
                              "MedWU": med_w_u, "MedWC": med_w_c})
        print(f"{mat:<12}  {n_obs_train:>12}  {str(auc):>6}  "
              f"{str(mae):>8}  {str(rmse):>8}  {str(r2):>6}  "
              f"{str(cov_u):>7}  {str(cov_c):>7}  {str(mean_w_u):>10}  "
              f"{str(mean_w_c):>10}  {str(med_w_u):>10}  {str(med_w_c):>10}")

    summary_df = pd.DataFrame(summary_rows)

    def _val(mat, col):
        v = summary_df.loc[summary_df["Material"] == mat, col].iat[0]
        return float(v) if not pd.isna(v) else float("nan")

    # â”€â”€ Figure 1 â€” Database coverage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    y_all_mask = np.vstack([data["y_train_mask"], data["y_val_mask"], data["y_test_mask"]])
    n_total    = len(y_all_mask)
    n_obs_all  = y_all_mask.sum(axis=0)
    pct        = n_obs_all / n_total * 100

    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(y_cols, n_obs_all, color=_C1, edgecolor="white", zorder=3)
    ax.axhline(n_total, color=_GRAY, lw=1.2, ls="--",
               label=f"Total buildings  ({n_total})", zorder=2)
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + n_total * 0.012,
                f"{p:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Material")
    ax.set_ylabel("Number of observed records")
    ax.set_title("Database coverage: observed records per material")
    ax.set_ylim(0, n_total * 1.20)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # â”€â”€ Figure 2 â€” Observation model AUC â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    aucs   = [_val(mat, "AUC") for mat in y_cols]
    colors = [_C1 if not pd.isna(v) else _GRAY for v in aucs]
    vals   = [v if not pd.isna(v) else 0.0 for v in aucs]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(y_cols, vals, color=colors, edgecolor="white", zorder=3)
    ax.axhline(0.5, color=_REF, lw=1.5, ls="--", label="Random baseline  (AUC = 0.5)", zorder=2)
    for bar, v in zip(bars, aucs):
        if not pd.isna(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Material")
    ax.set_ylabel("AUC-ROC (test set)")
    ax.set_title("Observation model performance by material")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # â”€â”€ Figure 3 â€” Conditional intensity MAE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    maes   = [_val(mat, "MAE") for mat in y_cols]
    colors = [_C1 if not pd.isna(v) else _GRAY for v in maes]
    vals   = [v if not pd.isna(v) else 0.0 for v in maes]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(y_cols, vals, color=colors, edgecolor="white", zorder=3)
    for bar, v in zip(bars, maes):
        if not pd.isna(v) and v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.012,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Material")
    ax.set_ylabel("MAE  (kg/mÂ²)")
    ax.set_title("Conditional reported-intensity model error by material")
    plt.tight_layout()
    plt.show()

    # â”€â”€ Figure 4 â€” Observed vs predicted â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig, axes = plt.subplots(1, len(y_cols), figsize=(14, 3.8))
    for ax, m in zip(axes, range(len(y_cols))):
        mat = y_cols[m]
        obs = data["y_test_mask"][:, m]
        if obs.sum() < 2:
            ax.text(0.5, 0.5, "insufficient\ndata", transform=ax.transAxes,
                    ha="center", va="center", color=_GRAY)
            ax.set_title(mat, fontsize=10)
            continue
        y_true = data["y_test_raw"][obs, m]
        y_pred = intervals_test[mat]["p50"][obs]
        use_log = (len(y_true) > 4 and
                   np.percentile(y_true, 95) / (np.percentile(y_true, 5) + 1e-6) > 10)
        ax.scatter(y_true, y_pred, s=18, alpha=0.5, color=_C1, edgecolors="none", zorder=3)
        lo = min(y_true.min(), y_pred.min()) * 0.85
        hi = max(y_true.max(), y_pred.max()) * 1.15
        if use_log:
            lo = max(lo, 1e-2)
        ax.plot([lo, hi], [lo, hi], color=_REF, lw=1.2, ls="--", label="1:1", zorder=2)
        if use_log:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(mat, fontsize=10)
        ax.set_xlabel("Observed  (kg/mÂ²)", fontsize=8)
        if m == 0:
            ax.set_ylabel("Predicted p50  (kg/mÂ²)", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Observed vs predicted reported intensity (test set)", fontsize=11)
    plt.tight_layout()
    plt.show()

    # â”€â”€ Figure 5 â€” Empirical coverage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    covs_u = [_val(mat, "CovU") for mat in y_cols]
    covs_c = [_val(mat, "CovC") for mat in y_cols]
    colors = [(_C1 if abs(v - 0.9) <= 0.1 else _C2) if not pd.isna(v) else _GRAY
              for v in covs_c]
    vals_u = [v if not pd.isna(v) else 0.0 for v in covs_u]
    vals_c = [v if not pd.isna(v) else 0.0 for v in covs_c]

    x = np.arange(len(y_cols))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - width / 2, vals_u, width, color=_GRAY, edgecolor="white",
           label="Uncalibrated", zorder=3, alpha=0.8)
    bars_c = ax.bar(x + width / 2, vals_c, width, color=colors, edgecolor="white",
                    label="Calibrated", zorder=3)
    ax.axhline(0.90, color=_REF, lw=1.5, ls="--", label="Nominal 90% coverage", zorder=2)
    for bar, v in zip(bars_c, covs_c):
        if not pd.isna(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.set_xticks(x); ax.set_xticklabels(y_cols)
    ax.set_xlabel("Material")
    ax.set_ylabel("Empirical coverage (test set)")
    ax.set_title("Empirical coverage of 90% prediction intervals")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # â”€â”€ Figure 6 â€” Example query â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    example_idx = next(
        (i for i in range(len(data["X_test_proc"])) if data["y_test_mask"][i].sum() >= 3), 0
    )
    X_ex   = data["X_test_proc"][example_idx : example_idx + 1]
    X_ex_raw = data["X_test_raw"].iloc[example_idx : example_idx + 1][X_cols]
    result = final_query_model.query(X_ex, X_raw=X_ex_raw)

    p_rec_ex = [float(result[mat]["p_recorded"][0])       for mat in y_cols]
    p50_ex   = [float(result[mat]["p50"][0])               for mat in y_cols]
    p05_ex   = [float(result[mat]["p05"][0])               for mat in y_cols]
    p95_ex   = [float(result[mat]["p95"][0])               for mat in y_cols]
    exp_rep  = [float(result[mat]["expected_reported"][0]) for mat in y_cols]
    x        = np.arange(len(y_cols))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(y_cols, p_rec_ex, color=_C1, edgecolor="white", zorder=3)
    for i, v in enumerate(p_rec_ex):
        ax1.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=9)
    ax1.set_ylim(0, 1.12); ax1.set_xlabel("Material")
    ax1.set_ylabel("Recording probability")
    ax1.set_title("P(recorded | building features)")

    err_lo = [max(p50_ex[i] - p05_ex[i], 0.0) for i in range(len(y_cols))]
    err_hi = [max(p95_ex[i] - p50_ex[i], 0.0) for i in range(len(y_cols))]
    ax2.bar(x, p50_ex, color=_C1, alpha=0.75, edgecolor="white",
            label="Conditional p50  (kg/mÂ²)", zorder=3)
    ax2.errorbar(x, p50_ex, yerr=[err_lo, err_hi],
                 fmt="none", color="black", capsize=5, lw=1.5, zorder=4)
    ax2.scatter(x, exp_rep, color=_C2, zorder=5, s=60,
                label="Expected reported  (p_rec Ã— p50)")
    ax2.set_xticks(x); ax2.set_xticklabels(y_cols)
    ax2.set_xlabel("Material"); ax2.set_ylabel("Reported intensity  (kg/mÂ²)")
    ax2.set_title("Conditional intensity + 90% prediction interval")
    ax2.legend(fontsize=8)
    fig.suptitle("Example query â€” single building from test set", fontsize=11)
    plt.tight_layout()
    plt.show()

    # â”€â”€ Save artifacts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    joblib.dump(final_query_model, "model.joblib")
    print("Saved: model.joblib")

    joblib.dump(data["preprocessor"], "preprocessor.joblib")
    print("Saved: preprocessor.joblib")

    with open("best_observation_params.json", "w") as _f:
        json.dump(best_observation_params, _f, indent=2)
    print("Saved: best_observation_params.json")

    with open("best_intensity_params.json", "w") as _f:
        json.dump(best_intensity_params, _f, indent=2)
    print("Saved: best_intensity_params.json")

    summary_df.to_csv("evaluation_summary.csv", index=False)
    print("Saved: evaluation_summary.csv")

    _model_info = {
        "X_cols":            X_cols,
        "y_cols":            y_cols,
        "SEED":              SEED,
        "LOW_OBS_THRESHOLD": LOW_OBS_THRESHOLD,
        "n_observed_train":  final_query_model.n_observed_train_,
        "data_split": {
            "n_train": int(len(data["X_train_proc"])),
            "n_val":   int(len(data["X_val_proc"])),
            "n_test":  int(len(data["X_test_proc"])),
        },
    }
    with open("model_info.json", "w") as _f:
        json.dump(_model_info, _f, indent=2)
    print("Saved: model_info.json")

    print("\nAll application files saved.")
    print("To run inference on new data:")
    print("  model = joblib.load('model.joblib')")
    print("  pre   = joblib.load('preprocessor.joblib')")
    print("  X_proc = pre.transform(X_raw[X_cols])")
    print("  result = model.query(X_proc)")


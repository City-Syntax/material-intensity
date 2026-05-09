import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from two_stage_model import GROUP_COLS, Y_COLS, TwoStageConditionalModel, build_group_keys

try:
    import optuna
except Exception:
    optuna = None

SEED = 42
MIN_OBSERVED_TARGETS = 2

X_COLS = ["Construction period", "Typology", "Primary Code", "Hybrid Structure", "Country"]


def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def prepare_data(
    file_path="Integrated_MI_database_add_Singapore.xlsx",
    clip_upper_quantile=0.99,
    clip_materials=("Steel", "Glass", "Concrete", "Brick", "Wood"),
    random_state=SEED,
):
    file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parent / file_path

    df = pd.read_excel(file_path)
    df["Construction period"] = pd.to_numeric(df["Construction period"], errors="coerce")
    df = df.dropna(subset=X_COLS).reset_index(drop=True)

    target_mask_df = df[Y_COLS].notna()
    df = df.loc[target_mask_df.sum(axis=1) >= MIN_OBSERVED_TARGETS].reset_index(drop=True)

    X = df[X_COLS].copy()
    y_df = df[Y_COLS].copy()
    y_mask = y_df.notna().to_numpy(dtype=bool)

    X_train, X_temp, y_train_df, y_temp_df, y_train_mask, y_temp_mask = train_test_split(
        X, y_df, y_mask, test_size=0.30, random_state=random_state
    )
    X_val, X_test, y_val_df, y_test_df, y_val_mask, y_test_mask = train_test_split(
        X_temp, y_temp_df, y_temp_mask, test_size=0.50, random_state=random_state
    )

    for df_ in [X_train, X_val, X_test, y_train_df, y_val_df, y_test_df]:
        df_.reset_index(drop=True, inplace=True)

    y_train_raw = y_train_df.to_numpy(dtype=np.float64)
    y_val_raw = y_val_df.to_numpy(dtype=np.float64)
    y_test_raw = y_test_df.to_numpy(dtype=np.float64)

    # Upper-quantile clipping computed on training data, applied to all splits
    clip_bounds = None
    if clip_upper_quantile is not None:
        mat_to_idx = {m: i for i, m in enumerate(Y_COLS)}
        upper_bounds = {}
        for mat in clip_materials:
            if mat not in mat_to_idx:
                continue
            idx = mat_to_idx[mat]
            obs = y_train_mask[:, idx]
            if not np.any(obs):
                continue
            ub = np.quantile(y_train_raw[obs, idx], clip_upper_quantile)
            for arr, mask in [
                (y_train_raw, y_train_mask),
                (y_val_raw, y_val_mask),
                (y_test_raw, y_test_mask),
            ]:
                rows = mask[:, idx]
                arr[rows, idx] = np.minimum(arr[rows, idx], ub)
            upper_bounds[mat] = float(ub)
        clip_bounds = {"upper_quantile": clip_upper_quantile, "upper_bounds": upper_bounds}

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["Construction period"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Typology", "Primary Code", "Hybrid Structure", "Country"],
            ),
        ]
    )
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    return {
        "preprocessor": preprocessor,
        "X_train_proc": X_train_proc,
        "X_val_proc": X_val_proc,
        "X_test_proc": X_test_proc,
        "X_train_raw": X_train,
        "X_val_raw": X_val,
        "X_test_raw": X_test,
        "y_train_raw": y_train_raw,
        "y_val_raw": y_val_raw,
        "y_test_raw": y_test_raw,
        "y_train_df": y_train_df,
        "y_val_df": y_val_df,
        "y_test_df": y_test_df,
        "y_train_mask": y_train_mask,
        "y_val_mask": y_val_mask,
        "y_test_mask": y_test_mask,
        "groups_train": build_group_keys(X_train),
        "groups_val": build_group_keys(X_val),
        "groups_test": build_group_keys(X_test),
        "clip_bounds": clip_bounds,
        "kept_rows": len(df),
    }


def compute_naive_maes(y_train_raw, y_train_mask):
    naive_maes = {}
    for m, mat in enumerate(Y_COLS):
        # Presence rows only: the conditional regressor is trained on y > 0,
        # so the naive baseline must be computed on the same support.
        obs = y_train_mask[:, m] & (y_train_raw[:, m] > 0)
        if obs.sum() > 0:
            vals = y_train_raw[obs, m]
            naive_maes[mat] = float(np.mean(np.abs(vals - vals.mean())))
    return naive_maes


def compute_mase(pred_dict, y_df, naive_maes, eps=1e-8):
    mases = []
    for mat in Y_COLS:
        y_true = y_df[mat].to_numpy(dtype=float)
        # Evaluate Stage 2 only on presence rows (observed AND positive).
        # Measuring conditional intensity predictions against y=0 rows penalises
        # Stage 2 for a job that belongs to Stage 1 (presence classification).
        obs = (~np.isnan(y_true)) & (y_true > 0)
        if np.any(obs):
            y_pred = pred_dict[mat]["p50"][obs]
            mae = np.mean(np.abs(y_pred - y_true[obs]))
            mases.append(mae / (naive_maes.get(mat, 1.0) + eps))
    return float(np.mean(mases)) if mases else np.inf


def fit_best_model(data, n_trials=0):
    x_train = data["X_train_proc"]
    x_val = data["X_val_proc"]

    naive_maes = compute_naive_maes(data["y_train_raw"], data["y_train_mask"])

    if n_trials <= 0 or optuna is None:
        model = TwoStageConditionalModel(random_state=SEED)
        model.fit(x_train, data["X_train_raw"], data["y_train_raw"], data["y_train_mask"])
        return model, {}

    def objective(trial):
        model = TwoStageConditionalModel(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 3, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            cov_shrink=trial.suggest_float("cov_shrink", 0.0, 1.0),
            min_group_size=trial.suggest_int("min_group_size", 10, 40),
            reg_eps=trial.suggest_float("reg_eps", 1e-6, 1e-2, log=True),
            random_state=SEED,
        )
        model.fit(x_train, data["X_train_raw"], data["y_train_raw"], data["y_train_mask"])
        pred_val = model.predict(x_val, data["groups_val"], alpha=0.10)
        return compute_mase(pred_val, data["y_val_df"], naive_maes)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params

    tuned = TwoStageConditionalModel(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        learning_rate=best["learning_rate"],
        subsample=best["subsample"],
        colsample_bytree=best["colsample_bytree"],
        reg_alpha=best["reg_alpha"],
        reg_lambda=best["reg_lambda"],
        cov_shrink=best["cov_shrink"],
        min_group_size=best["min_group_size"],
        reg_eps=best["reg_eps"],
        random_state=SEED,
    )
    tuned.fit(x_train, data["X_train_raw"], data["y_train_raw"], data["y_train_mask"])
    return tuned, best


def export_artifacts(model, preprocessor, best_params=None, out_dir="."):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, out / "preprocessor.joblib")
    joblib.dump(model, out / "model.joblib")

    model_info = {
        "model_type": "TwoStageConditionalModel",
        "stage1": "XGBClassifier per-material (binary:logistic)",
        "stage2": "MoE: GaussianMixture regime labels + XGBClassifier gating + XGBRegressor experts (log-space)",
        "stage2_n_components": model.stage2.n_components,
        "stage2_regime_counts": {
            mat: (model.stage2.models_[mat].K_ if model.stage2.models_.get(mat) is not None else 0)
            for mat in Y_COLS
        },
        "joint_layer": "MultivariateNormal with group-specific covariance (residual inspection only)",
        "group_cols": list(GROUP_COLS),
        "y_cols": list(Y_COLS),
        "X_cols": list(X_COLS),
        "stage1_xgb_params": model.stage1.xgb_params,
        "stage2_xgb_params": model.stage2.xgb_params,
    }
    if best_params:
        model_info["best_params"] = best_params

    with open(out / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/export TwoStageConditionalModel artifacts.")
    parser.add_argument("--data", default="Integrated_MI_database_add_Singapore.xlsx")
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    set_global_seed(SEED)
    data = prepare_data(file_path=args.data, random_state=SEED)
    model, best_params = fit_best_model(data, n_trials=args.trials)
    export_artifacts(model, data["preprocessor"], best_params=best_params, out_dir=args.out_dir)
    print("Saved: preprocessor.joblib, model.joblib, model_info.json")

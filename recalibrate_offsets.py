#!/usr/bin/env python
"""Recompute per-material conformal calibration offsets (delta_m) on the
validation split, persist them onto the already-trained FinalQueryModel
object, and re-export model_finalquery.joblib + model_info.json.

This does NOT retrain Stage 1 / Stage 2 models. It reuses the existing
fitted model_finalquery.joblib and only attaches/refreshes the
`validation_offsets_` attribute that query() applies to widen [p05, p95]
into a calibrated interval, so that the deployed model's query() output
matches the manuscript's description of the released tool.

Run from the material-intensity-main directory:
    python recalibrate_offsets.py
"""

import sys
import types
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import model_classes as mc

SEED = 42
SPLIT_MANIFEST_PATH = Path("data/processed/fixed_split_manifest.csv")
MIN_OBSERVED_TARGETS = 2
CAL_ALPHA = 0.10

PERIOD_BUCKETS = ["pre_1945", "1945_1970", "1970_1990", "1990_2010", "post_2010"]
X_cols = [
    "Construction period",
    "Construction period bucket",
    "Typology",
    "Primary Code",
    "Hybrid Structure",
    "Country",
    "Geo_macro",
]
y_cols = ["Concrete", "Glass", "Steel", "Wood", "Brick"]


def to_period_bucket(year_series):
    year = pd.to_numeric(year_series, errors="coerce")
    return pd.cut(
        year,
        bins=[-np.inf, 1945, 1970, 1990, 2010, np.inf],
        labels=PERIOD_BUCKETS,
        right=False,
    )


def prepare_data(
    file_path="data/processed/Integrated_MI_database_add_Singapore.xlsx",
    split_manifest_path=SPLIT_MANIFEST_PATH,
    min_observed_targets=MIN_OBSERVED_TARGETS,
):
    """Verbatim port of prepare_data() from prediction_model.py (data-loading
    and preprocessing only; no model fitting), kept in sync with that
    function so the val split reproduces exactly."""
    file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path

    df = pd.read_excel(file_path)
    df["ID_marked"] = df["ID_marked"].astype("string").str.strip()
    if df["ID_marked"].isna().any() or df["ID_marked"].duplicated().any():
        raise ValueError("ID_marked must be present and unique before applying the fixed split manifest.")
    df["Construction period"] = pd.to_numeric(df["Construction period"], errors="coerce")
    df["Construction period bucket"] = to_period_bucket(df["Construction period"])
    df = df.dropna(subset=X_cols).reset_index(drop=True)

    target_mask_df = df[y_cols].notna()
    df = df.loc[target_mask_df.sum(axis=1) >= min_observed_targets].reset_index(drop=True)

    split_manifest_path = Path(split_manifest_path)
    if not split_manifest_path.is_absolute():
        split_manifest_path = Path.cwd() / split_manifest_path
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Fixed split manifest not found: {split_manifest_path}.")
    manifest = pd.read_csv(split_manifest_path, dtype={"ID_marked": "string"})
    required_manifest_cols = {"ID_marked", "split_group_id", "split"}
    missing_manifest_cols = required_manifest_cols.difference(manifest.columns)
    if missing_manifest_cols:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing_manifest_cols)}")
    if manifest["ID_marked"].duplicated().any():
        raise ValueError("Split manifest contains duplicate ID_marked values.")

    split_by_id = manifest.set_index("ID_marked")["split"]
    df["_split"] = df["ID_marked"].map(split_by_id)
    missing_ids = df.loc[df["_split"].isna(), "ID_marked"].tolist()
    if missing_ids:
        raise ValueError(f"Fixed split manifest is missing {len(missing_ids)} current model IDs.")
    invalid_splits = set(df["_split"]) - {"train", "val", "test"}
    if invalid_splits:
        raise ValueError(f"Invalid split labels in manifest: {sorted(invalid_splits)}")

    current_groups = manifest.loc[manifest["ID_marked"].isin(df["ID_marked"])]
    crossed_groups = current_groups.groupby("split_group_id")["split"].nunique()
    if (crossed_groups > 1).any():
        raise ValueError("Split manifest places at least one split group in multiple splits.")

    def _take(split_name):
        part = df.loc[df["_split"] == split_name]
        X_part = part[X_cols].copy().reset_index(drop=True)
        y_part = part[y_cols].copy().reset_index(drop=True)
        mask_part = y_part.notna().to_numpy(dtype=bool)
        ids_part = part["ID_marked"].astype(str).reset_index(drop=True)
        return X_part, y_part, mask_part, ids_part

    X_train, y_train_df, y_train_mask, train_ids = _take("train")
    X_val, y_val_df, y_val_mask, val_ids = _take("val")
    X_test, y_test_df, y_test_mask, test_ids = _take("test")

    y_train_raw = y_train_df.to_numpy(dtype=np.float64).copy()
    y_val_raw   = y_val_df.to_numpy(dtype=np.float64).copy()
    y_test_raw  = y_test_df.to_numpy(dtype=np.float64).copy()

    for m in range(len(y_cols)):
        obs = y_train_mask[:, m]
        if obs.sum() < 2:
            continue
        cap = np.nanpercentile(y_train_raw[obs, m], 99.5)
        y_train_raw[obs, m] = np.minimum(y_train_raw[obs, m], cap)

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), ["Construction period"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["Construction period bucket", "Typology", "Primary Code", "Hybrid Structure", "Country", "Geo_macro"]),
    ])
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc   = preprocessor.transform(X_val)
    X_test_proc  = preprocessor.transform(X_test)

    return dict(
        X_train_proc=X_train_proc, X_val_proc=X_val_proc, X_test_proc=X_test_proc,
        X_train_raw=X_train,       X_val_raw=X_val,        X_test_raw=X_test,
        y_train_raw=y_train_raw,   y_val_raw=y_val_raw,    y_test_raw=y_test_raw,
        y_train_mask=y_train_mask, y_val_mask=y_val_mask,  y_test_mask=y_test_mask,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        preprocessor=preprocessor,
        kept_rows=len(df),
        split_manifest_path=split_manifest_path,
        split_manifest_sha256=hashlib.sha256(split_manifest_path.read_bytes()).hexdigest(),
    )


def _conformal_offset(y_true, p05, p95, alpha=CAL_ALPHA):
    """Verbatim port of _conformal_offset() from prediction_model.py."""
    scores = np.maximum(p05 - y_true, y_true - p95)
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 0.0
    k = int(np.ceil((scores.size + 1) * (1.0 - alpha)))
    k = max(1, min(k, scores.size))
    return float(np.sort(scores)[k - 1])


def load_existing_model(art_dir):
    """Load model_finalquery.joblib via model_classes definitions, regardless
    of what module name the existing pickle was saved under (bridges both
    the legacy 'prediction_model' registration and a native model_classes one)."""
    fake_pm = types.ModuleType("prediction_model")
    fake_pm.ObservationModel = mc.ObservationModel
    fake_pm.IntensityModel = mc.IntensityModel
    fake_pm.FinalQueryModel = mc.FinalQueryModel
    sys.modules.setdefault("prediction_model", fake_pm)
    return joblib.load(art_dir / "model_finalquery.joblib")


def main():
    art_dir = Path("artifacts")
    print("Loading existing trained model (no retraining) ...")
    final_query_model = load_existing_model(art_dir)
    print(f"  Loaded: {type(final_query_model)}")

    print("Reproducing data split (train/val/test) ...")
    data = prepare_data()
    print(f"  X_val_proc: {data['X_val_proc'].shape}")

    intervals_val = final_query_model.tuned_intensity_model.predict_quantiles(data["X_val_proc"])

    validation_offsets = {}
    n_val_obs = {}
    for m, mat in enumerate(y_cols):
        obs = data["y_val_mask"][:, m]
        n_val_obs[mat] = int(obs.sum())
        if obs.sum() < 2:
            validation_offsets[mat] = 0.0
            continue
        y_true = data["y_val_raw"][obs, m]
        p05    = intervals_val[mat]["p05"][obs]
        p95    = intervals_val[mat]["p95"][obs]
        validation_offsets[mat] = _conformal_offset(y_true, p05, p95)

    print()
    print(f"{'Material':<12}  {'n_val':>6}  {'delta_m':>10}")
    print("-" * 32)
    for mat in y_cols:
        print(f"{mat:<12}  {n_val_obs[mat]:>6}  {validation_offsets[mat]:>10.3f}")

    final_query_model.validation_offsets_ = validation_offsets

    # Sanity check: query() must now apply the offset and floor p05 at 0.
    probe = final_query_model.query(data["X_val_proc"][:5], X_raw=data["X_val_raw"].iloc[:5])
    for mat in y_cols:
        p05_uncal = intervals_val[mat]["p05"][:5]
        p05_cal   = probe[mat]["p05"]
        expected  = np.maximum(p05_uncal - validation_offsets[mat], 0.0)
        assert np.allclose(p05_cal, expected), f"query() offset mismatch for {mat}"
    print("\nSanity check passed: query() applies delta_m with p05 floored at 0.")

    art_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_query_model, art_dir / "model_finalquery.joblib")
    print(f"\nSaved: {art_dir / 'model_finalquery.joblib'}")
    print(f"  Pickled class module: {type(final_query_model).__module__}")

    info_path = art_dir / "model_info.json"
    model_info = json.loads(info_path.read_text())
    model_info["validation_offsets_delta_m"] = validation_offsets
    model_info["calibration_note"] = (
        "delta_m computed on the validation split via split-conformal "
        "calibration (nominal 90% target, CAL_ALPHA=0.10); query() returns "
        "[max(p05 - delta_m, 0), p95 + delta_m] as the calibrated interval."
    )
    info_path.write_text(json.dumps(model_info, indent=2))
    print(f"Saved: {info_path} (added validation_offsets_delta_m)")


if __name__ == "__main__":
    main()

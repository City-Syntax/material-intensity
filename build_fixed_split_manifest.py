#!/usr/bin/env python
"""Build or update the immutable ID-based model split manifest.

On first creation, unique retained records are assigned to a deterministic
70/15/15 train/validation/test split. On later runs, all existing assignments
are preserved and newly added records enter training.
"""

from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "Integrated_MI_database_add_Singapore.xlsx"
MANIFEST_PATH = BASE_DIR / "data" / "processed" / "fixed_split_manifest.csv"
SEED = 42
MIN_OBSERVED_TARGETS = 2
X_COLS = [
    "Construction period", "Typology", "Primary Code", "Hybrid Structure",
    "Country", "Geo_macro",
]
Y_COLS = ["Concrete", "Glass", "Steel", "Wood", "Brick"]
VALID_SPLITS = {"train", "val", "test"}


def load_model_rows(data_path=DATA_PATH):
    df = pd.read_excel(data_path)
    required = {"ID_marked", *X_COLS, *Y_COLS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["ID_marked"] = df["ID_marked"].astype("string").str.strip()
    if df["ID_marked"].isna().any() or (df["ID_marked"] == "").any():
        raise ValueError("ID_marked contains missing or blank values.")
    duplicates = df.loc[df["ID_marked"].duplicated(keep=False), "ID_marked"].unique()
    if len(duplicates):
        raise ValueError(f"ID_marked must be unique; examples: {duplicates[:10].tolist()}")

    df["Construction period"] = pd.to_numeric(df["Construction period"], errors="coerce")
    df = df.dropna(subset=X_COLS).copy()
    df["n_observed_targets"] = df[Y_COLS].notna().sum(axis=1)
    df = df.loc[df["n_observed_targets"] >= MIN_OBSERVED_TARGETS].copy()
    df["source_prefix"] = df["ID_marked"].str.split("-", n=1).str[0]
    return df.reset_index(drop=True)


def initial_group_split(df, random_state=SEED):
    indices = df.index.to_numpy()
    groups = df["split_group_id"].to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(outer.split(indices, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    val_rel, test_rel = next(inner.split(temp_idx, groups=groups[temp_idx]))

    split = pd.Series(index=df.index, dtype="string")
    split.iloc[train_idx] = "train"
    split.iloc[temp_idx[val_rel]] = "val"
    split.iloc[temp_idx[test_rel]] = "test"
    return split


def update_from_existing(df, manifest_path):
    existing = pd.read_csv(manifest_path, dtype={"ID_marked": "string"})
    required = {"ID_marked", "split"}
    missing = required.difference(existing.columns)
    if missing:
        raise ValueError(f"Existing manifest is missing columns: {sorted(missing)}")
    if existing["ID_marked"].duplicated().any():
        raise ValueError("Existing manifest contains duplicate ID_marked values.")
    invalid = set(existing["split"].dropna()) - VALID_SPLITS
    if invalid:
        raise ValueError(f"Existing manifest contains invalid splits: {sorted(invalid)}")

    previous = existing.set_index("ID_marked")["split"].to_dict()
    return df["ID_marked"].map(previous).fillna("train").astype("string")


def validate_manifest(df):
    if df["split"].isna().any() or set(df["split"]) != VALID_SPLITS:
        raise ValueError(f"Expected populated train/val/test splits; found {sorted(set(df['split']))}")
    if df["ID_marked"].duplicated().any():
        raise ValueError("The split manifest must contain one row per unique ID_marked.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--rebuild", action="store_true",
                        help="Discard existing assignments and define a new benchmark.")
    args = parser.parse_args()

    df = load_model_rows(args.data)
    # The retained dataset has already been deduplicated. Each ID is therefore
    # its own stable split unit; historical candidate-pair audits are not reused.
    df["split_group_id"] = "record:" + df["ID_marked"]
    if args.output.exists() and not args.rebuild:
        df["split"] = update_from_existing(df, args.output)
        action = "Updated"
    else:
        df["split"] = initial_group_split(df)
        action = "Created"
    validate_manifest(df)

    manifest = df[[
        "ID_marked", "split_group_id", "split", "source_prefix", "n_observed_targets"
    ]].copy()
    manifest.insert(0, "manifest_version", 1)
    manifest = manifest.sort_values(["split", "split_group_id", "ID_marked"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    print(f"{action}: {args.output}")
    print(f"Model rows: {len(manifest)}")
    print(manifest["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    print("Multi-record split groups:",
          int((manifest.groupby("split_group_id").size() > 1).sum()))
    print("Observed targets by split:")
    joined = df[["ID_marked", *Y_COLS]].merge(
        manifest[["ID_marked", "split"]], on="ID_marked", validate="one_to_one"
    )
    print(joined.groupby("split")[Y_COLS].count().reindex(["train", "val", "test"]).to_string())


if __name__ == "__main__":
    main()

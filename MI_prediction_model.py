from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
MIN_OBSERVED_TARGETS = 2


X_cols = [
    "Construction period",
    "Typology",
    "Primary Code",
    "Hybrid Structure",
    "Country",
]

y_cols = ["Concrete", "Glass", "Steel", "Wood", "Brick"]


class JointQuantileNet(nn.Module):
    def __init__(self, input_dim: int, M: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.M = M

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 3) for _ in range(M)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        raw = torch.stack([head(features) for head in self.heads], dim=1)

        q50 = raw[:, :, 0]
        delta_low = F.softplus(raw[:, :, 1]) + 1e-4
        delta_high = F.softplus(raw[:, :, 2]) + 1e-4

        q5 = q50 - delta_low
        q95 = q50 + delta_high

        return torch.stack([q5, q50, q95], dim=-1)


def quantile_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
) -> torch.Tensor:
    q_levels = y_true.new_tensor([0.05, 0.50, 0.95])
    errors = y_true.unsqueeze(-1) - y_pred
    pinball = torch.where(errors >= 0, q_levels * errors, (q_levels - 1) * errors)
    mask = y_mask.unsqueeze(-1).float()
    masked_sum = (pinball * mask).sum()
    n_observed = mask.sum() * len(q_levels)
    return masked_sum / (n_observed + 1e-8)


def prepare_dataloaders(
    file_path: str | Path = "Integrated_MI_database.xlsx",
    batch_size: int = 64,
    random_state: int = SEED,
    clip_upper_quantile: float | None = 0.99,
    clip_materials: tuple[str, ...] = ("Steel", "Glass", "Concrete", "Brick", "Wood"),
    min_observed_targets: int = MIN_OBSERVED_TARGETS,
):
    file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parent / file_path

    df = pd.read_excel(file_path)

    df["Construction period"] = pd.to_numeric(
        df["Construction period"], errors="coerce"
    )
    df = df.dropna(subset=X_cols).reset_index(drop=True)
    target_mask_df = df[y_cols].notna()
    df = df.loc[target_mask_df.sum(axis=1) >= min_observed_targets].reset_index(drop=True)

    X = df[X_cols].copy()
    y_raw_df = df[y_cols].copy()
    y_mask = y_raw_df.notna().to_numpy(dtype=bool)

    split_data = train_test_split(
        X, y_raw_df, y_mask, test_size=0.30, random_state=random_state
    )
    X_train, X_temp, y_train_raw_df, y_temp_raw_df, y_train_mask, y_temp_mask = split_data

    split_temp = train_test_split(
        X_temp, y_temp_raw_df, y_temp_mask, test_size=0.50, random_state=random_state
    )
    X_val, X_test, y_val_raw_df, y_test_raw_df, y_val_mask, y_test_mask = split_temp

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train_raw_df = y_train_raw_df.reset_index(drop=True)
    y_val_raw_df = y_val_raw_df.reset_index(drop=True)
    y_test_raw_df = y_test_raw_df.reset_index(drop=True)

    y_train_raw = y_train_raw_df.to_numpy(dtype=np.float32)
    y_val_raw = y_val_raw_df.to_numpy(dtype=np.float32)
    y_test_raw = y_test_raw_df.to_numpy(dtype=np.float32)

    clip_bounds = None
    if clip_upper_quantile is not None:
        if not (0.0 <= clip_upper_quantile <= 1.0):
            raise ValueError("clip_upper_quantile must satisfy 0 <= q <= 1")

        material_to_idx = {material: idx for idx, material in enumerate(y_cols)}
        selected_materials = [
            material for material in clip_materials if material in material_to_idx
        ]

        upper_bounds = {}
        for material in selected_materials:
            idx = material_to_idx[material]
            observed_train = y_train_mask[:, idx]
            if not np.any(observed_train):
                continue

            upper_bound = np.quantile(
                y_train_raw[observed_train, idx], clip_upper_quantile
            )

            for target_array, target_mask in (
                (y_train_raw, y_train_mask),
                (y_val_raw, y_val_mask),
                (y_test_raw, y_test_mask),
            ):
                observed_rows = target_mask[:, idx]
                target_array[observed_rows, idx] = np.minimum(
                    target_array[observed_rows, idx], upper_bound
                )

            upper_bounds[material] = float(upper_bound)

        clip_bounds = {
            "upper_quantile": clip_upper_quantile,
            "upper_bounds": upper_bounds,
        }

    y_train_filled = np.nan_to_num(y_train_raw, nan=0.0)
    y_val_filled = np.nan_to_num(y_val_raw, nan=0.0)
    y_test_filled = np.nan_to_num(y_test_raw, nan=0.0)

    y_train = np.where(y_train_mask, np.log1p(y_train_filled), 0.0).astype(np.float32)
    y_val = np.where(y_val_mask, np.log1p(y_val_filled), 0.0).astype(np.float32)
    y_test = np.where(y_test_mask, np.log1p(y_test_filled), 0.0).astype(np.float32)

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

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_train_mask_tensor = torch.tensor(y_train_mask, dtype=torch.bool)
    y_val_mask_tensor = torch.tensor(y_val_mask, dtype=torch.bool)
    y_test_mask_tensor = torch.tensor(y_test_mask, dtype=torch.bool)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, y_train_mask_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor, y_val_mask_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, y_test_mask_tensor)

    loader_kwargs = {"pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return {
        "X_train": X_train_tensor,
        "X_val": X_val_tensor,
        "X_test": X_test_tensor,
        "y_train": y_train_tensor,
        "y_val": y_val_tensor,
        "y_test": y_test_tensor,
        "y_train_mask": y_train_mask_tensor,
        "y_val_mask": y_val_mask_tensor,
        "y_test_mask": y_test_mask_tensor,
        "y_val_raw_df": y_val_raw_df,
        "y_test_raw_df": y_test_raw_df,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "preprocessor": preprocessor,
        "clip_bounds": clip_bounds,
        "kept_rows": len(df),
        "min_observed_targets": min_observed_targets,
    }


if __name__ == "__main__":
    data = prepare_dataloaders()
    print("Data preparation complete.")
    print(f"X_train shape: {data['X_train'].shape}, y_train shape: {data['y_train'].shape}")
    print(f"X_val shape:   {data['X_val'].shape}, y_val shape:   {data['y_val'].shape}")
    print(f"X_test shape:  {data['X_test'].shape}, y_test shape:  {data['y_test'].shape}")
    print(
        f"Rows kept with >= {data['min_observed_targets']} observed targets: {data['kept_rows']}"
    )

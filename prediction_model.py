import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
MIN_OBSERVED_TARGETS = 2
QUANTILES = [0.05, 0.50, 0.95]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


X_cols = [
    "Construction period",
    "Typology",
    "Primary Code",
    "Hybrid Structure",
    "Country",
]

y_cols = ["Concrete", "Glass", "Steel", "Wood", "Brick"]


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def reset_run_seed(seed: int = SEED):
    """Reset all RNGs before each major run stage (tuning, training, evaluation)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_global_seed(seed)


reset_run_seed(SEED)


def split_inputs(
    x_full: torch.Tensor,
    structure_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if structure_dim <= 0 or structure_dim >= x_full.shape[1]:
        raise ValueError(
            f"Invalid structure_dim={structure_dim} for input width={x_full.shape[1]}"
        )
    x_all = x_full[:, :-structure_dim]
    x_structure = x_full[:, -structure_dim:]
    return x_all, x_structure


class JointQuantileNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        structure_dim: int,
        M: int = 5,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.M = M
        self.structure_dim = structure_dim
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.base_feature = nn.Linear(hidden_dim + structure_dim, hidden_dim)
        self.material_embeddings = nn.Parameter(torch.randn(M, hidden_dim) * 0.02)

        self.msg_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 3) for _ in range(M)])

    def forward(self, x_all: torch.Tensor, x_structure: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x_all)
        h_concat = torch.cat([h, x_structure], dim=1)

        base = self.base_feature(h_concat)
        z0 = base.unsqueeze(1) + self.material_embeddings.unsqueeze(0)

        q = self.msg_query(z0)
        k = self.msg_key(z0)
        v = self.msg_value(z0)
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.hidden_dim)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        msg = torch.matmul(attn_weights, v)

        gate = torch.sigmoid(self.msg_gate(torch.cat([z0, msg], dim=-1)))
        z = gate * msg + (1.0 - gate) * z0

        raw = torch.stack([head(z[:, m, :]) for m, head in enumerate(self.heads)], dim=1)

        q50 = raw[:, :, 0]
        d_q5 = F.softplus(raw[:, :, 1]) + 1e-4
        d_q95 = F.softplus(raw[:, :, 2]) + 1e-4
        q5 = q50 - d_q5
        q95 = q50 + d_q95

        return torch.stack([q5, q50, q95], dim=-1)


def quantile_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
) -> torch.Tensor:
    q_levels = y_true.new_tensor(QUANTILES)
    errors = y_true.unsqueeze(-1) - y_pred
    pinball = torch.where(errors >= 0, q_levels * errors, (q_levels - 1) * errors)
    mask = y_mask.unsqueeze(-1).float()
    masked_sum = (pinball * mask).sum()
    n_observed = mask.sum() * len(q_levels)
    return masked_sum / (n_observed + 1e-8)


def _pinball(pred_q: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
    e = target - pred_q
    return torch.maximum((q - 1) * e, q * e)


def compute_tail_weights(
    y: torch.Tensor,
    mask: torch.Tensor,
    tail_weight_power: float = 1.5,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Compute per-sample tail weights based on magnitude within each material.
    
    Weight formula: w = 1 + alpha * x^p, where x is normalized to [0, 1].
    This ensures baseline weight of 1.0 for low-value samples while emphasizing high values.
    Weights are normalized to unit mean per material to preserve overall loss scale.
    
    Args:
        y: (B, M) target tensor
        mask: (B, M) bool mask for observed values
        tail_weight_power: float, exponent for weight computation (e.g., 1.5 or 2.0)
        alpha: float, scaling factor for the power term (default 1.0)
        
    Returns:
        weights: (B, M) float tensor with normalized per-sample weights
    """
    B, M = y.shape
    weights = torch.ones_like(y)
    
    for m in range(M):
        m_mask = mask[:, m]
        if m_mask.any():
            y_m = y[m_mask, m]
            # Normalize to [0, 1] range: (y - min) / (max - min + eps)
            y_min = y_m.min()
            y_max = y_m.max()
            y_normalized = (y_m - y_min) / (y_max - y_min + 1e-6)
            # Weight formula: 1 + alpha * x^p
            # Ensures baseline weight of 1.0, emphasizes high values
            material_weights = 1.0 + alpha * torch.pow(y_normalized, tail_weight_power)
            # Normalize to unit mean per material
            material_weights = material_weights / (material_weights.mean() + 1e-8)
            weights[m_mask, m] = material_weights
    
    return weights


def masked_multi_quantile_loss(
    preds: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    quantiles: list[float] = QUANTILES,
    use_tail_weights: bool = True,
    tail_weight_power: float = 1.5,
) -> torch.Tensor:
    """
    Masked pinball loss with optional tail sample weighting applied only to q95.
    
    Args:
        preds: (B, M, Q) predicted quantiles
        y: (B, M) target values (in transformed space)
        mask: (B, M) bool mask for observed values
        quantiles: list of quantile levels
        use_tail_weights: if True, apply magnitude-based weighting only to q95 to improve upper tail coverage
        tail_weight_power: exponent for weight computation (higher = more emphasis on tails)
    
    Returns:
        scalar loss
    """
    B, M, Q = preds.shape
    losses = []
    valid_count = 0

    for m in range(M):
        m_mask = mask[:, m]
        if m_mask.any():
            y_m = y[m_mask, m]
            pred_m = preds[m_mask, m, :]
            
            lq = 0.0
            for qi, q in enumerate(quantiles):
                pinball_losses = _pinball(pred_m[:, qi], y_m, q)  # (N,)
                weighted_loss = pinball_losses.mean()
                lq = lq + weighted_loss
            
            losses.append(lq / Q)
            valid_count += 1

    if valid_count == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    return torch.stack(losses).mean()


def train_one_epoch(
    model: JointQuantileNet,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    structure_dim: int,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for X_batch, y_batch, m_batch in loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)
        m_batch = m_batch.to(DEVICE, non_blocking=True)

        x_all, x_structure = split_inputs(X_batch, structure_dim)
        optimizer.zero_grad(set_to_none=True)
        preds = model(x_all, x_structure)
        loss = masked_multi_quantile_loss(preds, y_batch, m_batch)
        loss.backward()
        optimizer.step()

        total += loss.item() * X_batch.size(0)
        n += X_batch.size(0)
    return total / max(n, 1)


def evaluate_loss(
    model: JointQuantileNet,
    loader: torch.utils.data.DataLoader,
    structure_dim: int,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for X_batch, y_batch, m_batch in loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)
            m_batch = m_batch.to(DEVICE, non_blocking=True)

            x_all, x_structure = split_inputs(X_batch, structure_dim)
            preds = model(x_all, x_structure)
            loss = masked_multi_quantile_loss(preds, y_batch, m_batch)

            total += loss.item() * X_batch.size(0)
            n += X_batch.size(0)
    return total / max(n, 1)


def fit_target_transformers(
    y_train_raw: np.ndarray,
    y_train_mask: np.ndarray,
) -> dict[str, QuantileTransformer | None]:
    transformers: dict[str, QuantileTransformer | None] = {}

    for idx, material in enumerate(y_cols):
        observed = y_train_mask[:, idx]
        if not np.any(observed):
            transformers[material] = None
            continue

        observed_targets = y_train_raw[observed, idx].reshape(-1, 1)
        transformer = QuantileTransformer(
            n_quantiles=min(1000, observed_targets.shape[0]),
            output_distribution="normal",
            random_state=SEED,
        )
        transformer.fit(observed_targets)
        transformers[material] = transformer

    return transformers


def transform_targets(
    y_raw: np.ndarray,
    y_mask: np.ndarray,
    target_transformers: dict[str, QuantileTransformer | None],
) -> np.ndarray:
    transformed = np.zeros_like(y_raw, dtype=np.float32)

    for idx, material in enumerate(y_cols):
        observed = y_mask[:, idx]
        if not np.any(observed):
            continue

        transformer = target_transformers.get(material)
        observed_targets = y_raw[observed, idx].reshape(-1, 1)
        if transformer is None:
            transformed[observed, idx] = observed_targets[:, 0].astype(np.float32)
            continue

        transformed[observed, idx] = transformer.transform(observed_targets)[:, 0].astype(
            np.float32
        )

    return transformed


def inverse_transform_quantiles(
    quantiles: np.ndarray,
    target_transformers: dict[str, QuantileTransformer | None] | None,
) -> np.ndarray:
    if target_transformers is None:
        return np.expm1(quantiles)

    restored = np.empty_like(quantiles, dtype=np.float32)
    for idx, material in enumerate(y_cols):
        transformer = target_transformers.get(material)
        material_quantiles = quantiles[:, idx, :]
        if transformer is None:
            restored[:, idx, :] = material_quantiles.astype(np.float32)
            continue

        for q_idx in range(material_quantiles.shape[1]):
            restored[:, idx, q_idx] = transformer.inverse_transform(
                material_quantiles[:, q_idx].reshape(-1, 1)
            )[:, 0].astype(np.float32)

    return restored


def prepare_dataloaders(
    file_path: str | Path = "Integrated_MI_database_add_Singapore.xlsx",
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

    # temp (30%) is split equally into validation, calibration, and test (10% each)
    split_temp_test = train_test_split(
        X_temp, y_temp_raw_df, y_temp_mask, test_size=1.0 / 3.0, random_state=random_state
    )
    (
        X_val_calib,
        X_test,
        y_val_calib_raw_df,
        y_test_raw_df,
        y_val_calib_mask,
        y_test_mask,
    ) = split_temp_test
    split_val_calib = train_test_split(
        X_val_calib,
        y_val_calib_raw_df,
        y_val_calib_mask,
        test_size=0.50,
        random_state=random_state,
    )
    X_val, X_calib, y_val_raw_df, y_calib_raw_df, y_val_mask, y_calib_mask = split_val_calib

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_calib = X_calib.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train_raw_df = y_train_raw_df.reset_index(drop=True)
    y_val_raw_df = y_val_raw_df.reset_index(drop=True)
    y_calib_raw_df = y_calib_raw_df.reset_index(drop=True)
    y_test_raw_df = y_test_raw_df.reset_index(drop=True)

    y_train_raw = np.array(y_train_raw_df.to_numpy(dtype=np.float32), copy=True)
    y_val_raw = np.array(y_val_raw_df.to_numpy(dtype=np.float32), copy=True)
    y_calib_raw = np.array(y_calib_raw_df.to_numpy(dtype=np.float32), copy=True)
    y_test_raw = np.array(y_test_raw_df.to_numpy(dtype=np.float32), copy=True)

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
                (y_calib_raw, y_calib_mask),
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
    y_calib_filled = np.nan_to_num(y_calib_raw, nan=0.0)
    y_test_filled = np.nan_to_num(y_test_raw, nan=0.0)

    target_transformers = fit_target_transformers(y_train_filled, y_train_mask)
    y_train = transform_targets(y_train_filled, y_train_mask, target_transformers)
    y_val = transform_targets(y_val_filled, y_val_mask, target_transformers)
    y_calib = transform_targets(y_calib_filled, y_calib_mask, target_transformers)
    y_test = transform_targets(y_test_filled, y_test_mask, target_transformers)

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
    X_calib_processed = preprocessor.transform(X_calib)
    X_test_processed = preprocessor.transform(X_test)

    structure_dim = len(preprocessor.named_transformers_["cat"].categories_[2])

    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
    X_calib_tensor = torch.tensor(X_calib_processed, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    y_calib_tensor = torch.tensor(y_calib, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_train_mask_tensor = torch.tensor(y_train_mask, dtype=torch.bool)
    y_val_mask_tensor = torch.tensor(y_val_mask, dtype=torch.bool)
    y_calib_mask_tensor = torch.tensor(y_calib_mask, dtype=torch.bool)
    y_test_mask_tensor = torch.tensor(y_test_mask, dtype=torch.bool)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, y_train_mask_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor, y_val_mask_tensor)
    calib_dataset = TensorDataset(X_calib_tensor, y_calib_tensor, y_calib_mask_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, y_test_mask_tensor)

    loader_kwargs = {"pin_memory": torch.cuda.is_available()}
    train_generator = torch.Generator().manual_seed(random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return {
        "X_train": X_train_tensor,
        "X_val": X_val_tensor,
        "X_calib": X_calib_tensor,
        "X_test": X_test_tensor,
        "y_train": y_train_tensor,
        "y_val": y_val_tensor,
        "y_calib": y_calib_tensor,
        "y_test": y_test_tensor,
        "y_train_mask": y_train_mask_tensor,
        "y_val_mask": y_val_mask_tensor,
        "y_calib_mask": y_calib_mask_tensor,
        "y_test_mask": y_test_mask_tensor,
        "y_val_raw_df": y_val_raw_df,
        "y_calib_raw_df": y_calib_raw_df,
        "y_test_raw_df": y_test_raw_df,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "calib_loader": calib_loader,
        "test_loader": test_loader,
        "preprocessor": preprocessor,
        "target_transformers": target_transformers,
        "clip_bounds": clip_bounds,
        "kept_rows": len(df),
        "min_observed_targets": min_observed_targets,
        "structure_dim": structure_dim,
    }


def predict_material_ranges(
    model: JointQuantileNet,
    X_new_tensor: torch.Tensor,
    structure_dim: int,
    target_transformers: dict[str, QuantileTransformer | None] | None = None,
    conformal_qhats: dict | None = None,
) -> dict:
    """
    Predict p5/p50/p95 material intensities in physical units (kg/m²).

    If conformal_qhats is provided, CQR correction is applied to the outer quantiles:
        p5_calibrated = max(p5 - qhat, 0)
        p95_calibrated = p95 + qhat
    """
    model.eval()
    with torch.no_grad():
        X_new = X_new_tensor.to(DEVICE)
        x_all, x_structure = split_inputs(X_new, structure_dim)
        q_pred = model(x_all, x_structure).cpu().numpy()  # (N, M, 3) in transformed space

    restored_quantiles = inverse_transform_quantiles(q_pred, target_transformers)
    q5 = restored_quantiles[:, :, 0]
    q50 = restored_quantiles[:, :, 1]
    q95 = restored_quantiles[:, :, 2]

    if conformal_qhats is not None:
        for m, material in enumerate(y_cols):
            qhat = float(conformal_qhats.get(material, 0.0))
            q5[:, m] = np.maximum(q5[:, m] - qhat, 0.0)
            q95[:, m] = q95[:, m] + qhat

    return {
        material: {
            "p5": q5[:, m],
            "p50": q50[:, m],
            "p95": q95[:, m],
        }
        for m, material in enumerate(y_cols)
    }


def fit_conformal_qhats(
    model: JointQuantileNet,
    X_calib_tensor: torch.Tensor,
    y_calib_raw_df: pd.DataFrame,
    alpha: float = 0.10,
    structure_dim: int = 1,
    target_transformers: dict[str, QuantileTransformer | None] | None = None,
) -> dict:
    """
    Split-conformal calibration using CQR nonconformity scores on a dedicated
    calibration set.

    scores_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)) in physical units.
    Scores may be negative when y_i lies inside [q_lo, q_hi].
    """
    base_pred = predict_material_ranges(
        model,
        X_calib_tensor,
        structure_dim=structure_dim,
        target_transformers=target_transformers,
    )
    qhats = {}

    for material in y_cols:
        y_obs = y_calib_raw_df[material].to_numpy(dtype=float)
        mask = ~np.isnan(y_obs)
        if mask.sum() == 0:
            qhats[material] = 0.0
            continue

        lo = base_pred[material]["p5"][mask]
        hi = base_pred[material]["p95"][mask]
        yv = y_obs[mask]

        scores = np.maximum(lo - yv, yv - hi)
        n = scores.shape[0]
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        qhats[material] = float(np.quantile(scores, q_level, method="higher"))

    return qhats


def evaluate_conformal_coverage(
    model: JointQuantileNet,
    X_tensor: torch.Tensor,
    y_raw_df: pd.DataFrame,
    qhats: dict,
    structure_dim: int,
    target_transformers: dict[str, QuantileTransformer | None] | None = None,
) -> pd.DataFrame:
    """Return per-material coverage and mean interval width after conformal correction."""
    pred = predict_material_ranges(
        model,
        X_tensor,
        structure_dim=structure_dim,
        target_transformers=target_transformers,
        conformal_qhats=qhats,
    )
    rows = []
    for material in y_cols:
        y_obs = y_raw_df[material].to_numpy(dtype=float)
        mask = ~np.isnan(y_obs)
        if mask.sum() == 0:
            rows.append({"material": material, "coverage": np.nan,
                         "mean_interval_width": np.nan, "n_obs": 0})
            continue

        lo = pred[material]["p5"][mask]
        hi = pred[material]["p95"][mask]
        yv = y_obs[mask]

        rows.append({
            "material": material,
            "coverage": float(np.mean((yv >= lo) & (yv <= hi))),
            "mean_interval_width": float(np.mean(hi - lo)),
            "n_obs": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def evaluate_mdn_intervals(
    model: JointQuantileNet,
    data: dict,
    conformal_qhats: dict | None = None,
    split: str = "test",
    structure_dim: int = 1,
    target_transformers: dict[str, QuantileTransformer | None] | None = None,
) -> pd.DataFrame:
    """Return per-material coverage, mean interval width, and MAE."""
    X = data[f"X_{split}"]
    y_true_df = data[f"y_{split}_raw_df"]
    pred = predict_material_ranges(
        model,
        X,
        structure_dim=structure_dim,
        target_transformers=target_transformers,
        conformal_qhats=conformal_qhats,
    )
    rows = []
    for material in y_cols:
        y_true = y_true_df[material].to_numpy(dtype=float)
        mask = ~np.isnan(y_true)
        if mask.sum() == 0:
            continue
        lo = pred[material]["p5"][mask]
        hi = pred[material]["p95"][mask]
        med = pred[material]["p50"][mask]
        yt = y_true[mask]
        rows.append({
            "material": material,
            "coverage": float(np.mean((yt >= lo) & (yt <= hi))),
            "mean_width": float(np.mean(hi - lo)),
            "mae": float(np.mean(np.abs(med - yt))),
        })
    return pd.DataFrame(rows)


def _gaussian_kl(
    mu_p: np.ndarray,
    cov_p: np.ndarray,
    mu_q: np.ndarray,
    cov_q: np.ndarray,
    eps: float = 1e-6,
) -> float:
    d = mu_p.shape[0]
    cov_p_reg = cov_p + np.eye(d) * eps
    cov_q_reg = cov_q + np.eye(d) * eps

    sign_p, logdet_p = np.linalg.slogdet(cov_p_reg)
    sign_q, logdet_q = np.linalg.slogdet(cov_q_reg)
    if sign_p <= 0 or sign_q <= 0:
        return float("nan")

    inv_q = np.linalg.inv(cov_q_reg)
    mean_term = float((mu_q - mu_p).T @ inv_q @ (mu_q - mu_p))
    trace_term = float(np.trace(inv_q @ cov_p_reg))
    return 0.5 * (trace_term + mean_term - d + (logdet_q - logdet_p))


def evaluate_joint_distribution(
    model: JointQuantileNet,
    data: dict,
    split: str = "test",
    structure_dim: int = 1,
    min_complete_rows: int = 30,
    target_transformers: dict[str, QuantileTransformer | None] | None = None,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Evaluate joint distribution alignment between true and predicted medians.

    Returns (summary_dict, per_material_df, corr_true, corr_pred).
    """
    y_df = data[f"y_{split}_raw_df"][y_cols].copy()
    complete_mask = y_df.notna().all(axis=1).to_numpy()
    n_complete = int(complete_mask.sum())

    if n_complete < min_complete_rows:
        raise ValueError(
            f"Not enough complete rows for joint evaluation: {n_complete} < {min_complete_rows}"
        )

    pred_ranges_full = predict_material_ranges(
        model,
        data[f"X_{split}"],
        structure_dim=structure_dim,
        target_transformers=target_transformers,
        conformal_qhats=None,
    )
    pred_med_full = np.column_stack([pred_ranges_full[m]["p50"] for m in y_cols])
    y_true_full = y_df.to_numpy(dtype=float)

    y_true = y_true_full[complete_mask]
    y_pred = pred_med_full[complete_mask]

    mu_true = y_true.mean(axis=0)
    mu_pred = y_pred.mean(axis=0)
    std_true = y_true.std(axis=0, ddof=1)
    std_pred = y_pred.std(axis=0, ddof=1)
    cov_true = np.cov(y_true, rowvar=False)
    cov_pred = np.cov(y_pred, rowvar=False)
    corr_true = np.corrcoef(y_true, rowvar=False)
    corr_pred = np.corrcoef(y_pred, rowvar=False)

    summary = {
        "split": split,
        "n_complete_rows": n_complete,
        "mean_vector_mae": float(np.mean(np.abs(mu_true - mu_pred))),
        "cov_fro_norm": float(np.linalg.norm(cov_true - cov_pred, ord="fro")),
        "corr_fro_norm": float(np.linalg.norm(corr_true - corr_pred, ord="fro")),
        "kl_true_to_pred": _gaussian_kl(mu_true, cov_true, mu_pred, cov_pred),
        "kl_pred_to_true": _gaussian_kl(mu_pred, cov_pred, mu_true, cov_true),
    }
    per_material = pd.DataFrame({
        "material": y_cols,
        "true_mean": mu_true,
        "pred_mean": mu_pred,
        "abs_mean_diff": np.abs(mu_true - mu_pred),
        "true_std": std_true,
        "pred_std": std_pred,
        "abs_std_diff": np.abs(std_true - std_pred),
    })
    return summary, per_material, corr_true, corr_pred


if __name__ == "__main__":
    reset_run_seed(SEED)
    data = prepare_dataloaders()
    print("Data preparation complete.")
    print(f"X_train shape: {data['X_train'].shape}, y_train shape: {data['y_train'].shape}")
    print(f"X_val shape:   {data['X_val'].shape}, y_val shape:   {data['y_val'].shape}")
    print(f"X_calib shape: {data['X_calib'].shape}, y_calib shape: {data['y_calib'].shape}")
    print(f"X_test shape:  {data['X_test'].shape}, y_test shape:  {data['y_test'].shape}")
    print(
        f"Rows kept with >= {data['min_observed_targets']} observed targets: {data['kept_rows']}"
    )
    for split in ["train", "val", "calib", "test"]:
        mask = data[f"y_{split}_mask"].cpu().numpy()
        observed_counts = mask.sum(axis=1)
        print(
            f"{split.title()} observed targets per row - "
            f"min: {observed_counts.min()}, "
            f"mean: {observed_counts.mean():.2f}, "
            f"max: {observed_counts.max()}"
        )

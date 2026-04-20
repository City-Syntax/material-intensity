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


X_cols = [
    "Construction period",
    "Typology",
    "Primary Code",
    "Hybrid Structure",
    "Location_code",
]

y_cols = ["Concrete", "Glass", "Steel", "Wood", "Brick"]


class JointMDN(nn.Module):
    def __init__(self, input_dim: int = 5, M: int = 5, K: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.M = M
        self.K = K
        self.num_cholesky_params = M * (M + 1) // 2

        output_dim = K + (K * M) + (K * self.num_cholesky_params)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def build_cholesky(self, L_params: torch.Tensor) -> torch.Tensor:
        batch_size = L_params.shape[0]
        L = L_params.new_zeros(batch_size, self.K, self.M, self.M)

        tril_rows, tril_cols = torch.tril_indices(
            row=self.M, col=self.M, offset=0, device=L_params.device
        )
        L[:, :, tril_rows, tril_cols] = L_params

        diag_idx = torch.arange(self.M, device=L_params.device)
        L[:, :, diag_idx, diag_idx] = (
            F.softplus(L[:, :, diag_idx, diag_idx]) + 1e-6
        )
        return L

    def forward(self, x: torch.Tensor):
        output = self.network(x)

        pi_end = self.K
        mu_end = pi_end + (self.K * self.M)

        pi_logits = output[:, :pi_end]
        mu = output[:, pi_end:mu_end].view(-1, self.K, self.M)
        L_params = output[:, mu_end:].view(-1, self.K, self.num_cholesky_params)
        L = self.build_cholesky(L_params)

        return pi_logits, mu, L


def mdn_loss(pi_logits: torch.Tensor, mu: torch.Tensor, L: torch.Tensor, y_true: torch.Tensor):
    component_distribution = torch.distributions.MultivariateNormal(
        loc=mu, scale_tril=L
    )
    mixture_distribution = torch.distributions.Categorical(logits=pi_logits)
    joint_distribution = torch.distributions.MixtureSameFamily(
        mixture_distribution, component_distribution
    )

    log_prob = joint_distribution.log_prob(y_true)
    return -log_prob.mean()


def prepare_dataloaders(
    file_path: str | Path = "Final_database_clean_rows_removed.xlsx",
    batch_size: int = 64,
    random_state: int = 42,
):
    file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parent / file_path

    df = pd.read_excel(file_path)

    # Ensure the numeric feature is properly typed and remove incomplete rows.
    df["Construction period"] = pd.to_numeric(
        df["Construction period"], errors="coerce"
    )
    df = df.dropna(subset=X_cols + y_cols).reset_index(drop=True)

    X = df[X_cols].copy()
    y = np.log1p(df[y_cols].to_numpy(dtype=np.float32))

    # 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["Construction period"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Typology", "Primary Code", "Hybrid Structure", "Location_code"],
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

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "X_train": X_train_tensor,
        "X_val": X_val_tensor,
        "X_test": X_test_tensor,
        "y_train": y_train_tensor,
        "y_val": y_val_tensor,
        "y_test": y_test_tensor,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "preprocessor": preprocessor,
    }


if __name__ == "__main__":
    data = prepare_dataloaders()
    print("Data preparation complete.")
    print(f"X_train shape: {data['X_train'].shape}, y_train shape: {data['y_train'].shape}")
    print(f"X_val shape:   {data['X_val'].shape}, y_val shape:   {data['y_val'].shape}")
    print(f"X_test shape:  {data['X_test'].shape}, y_test shape:  {data['y_test'].shape}")

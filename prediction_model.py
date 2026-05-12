import contextlib
import io
import os
import random
import warnings
from pathlib import Path

import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import optuna
from scipy import stats
from scipy.stats import probplot, shapiro, kstest, norm as _norm, wasserstein_distance
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

SEED = 42
MIN_OBSERVED_TARGETS = 2
GROUP_COLS = ["Primary Code"]
LOG_EPS = 1e-6

X_cols = ["Construction period", "Typology", "Primary Code", "Hybrid Structure", "Country"]
y_cols = ["Concrete", "Glass", "Steel", "Wood", "Brick"]

# ── Sampling hyperparameters ────────────────────────────────────────────────
BLEND_LAMBDA   = 0.35   # chain weight in blend:  p = λ·p_chain + (1-λ)·p_marginal
DROPOUT_RATE   = 0.08   # diversity dropout: flip "present" → "absent" with this prob
P_CLIP_LO      = 0.05   # presence probability hard lower clip
P_CLIP_HI      = 0.95   # presence probability hard upper clip
RESIDUAL_SCALE = 0.35   # Stage 3 log-space residual amplitude
CROSS_DAMP     = 0.25   # Stage 3 off-diagonal covariance damping factor
Z_QUANT_95     = 1.6449 # norm.ppf(0.95); recovers sigma from [p05, p95] spread


def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


# ==========================================================
# Data Preparation
# ==========================================================

def build_group_keys(df, group_cols=GROUP_COLS):
    return (
        df.loc[:, group_cols]
        .fillna("Missing")
        .astype(str)
        .agg(" | ".join, axis=1)
        .to_numpy()
    )


def prepare_data(
    file_path="Integrated_MI_database_add_Singapore.xlsx",
    test_size=0.30,
    val_size=0.50,
    clip_upper_quantile=0.99,
    clip_materials=("Steel", "Glass", "Concrete", "Brick", "Wood"),
    min_observed_targets=MIN_OBSERVED_TARGETS,
    random_state=SEED,
):
    file_path = Path(file_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path

    df = pd.read_excel(file_path)
    df["Construction period"] = pd.to_numeric(df["Construction period"], errors="coerce")
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

    y_train_raw = y_train_df.to_numpy(dtype=np.float64)
    y_val_raw   = y_val_df.to_numpy(dtype=np.float64)
    y_test_raw  = y_test_df.to_numpy(dtype=np.float64)

    clip_bounds = None
    if clip_upper_quantile is not None:
        mat_to_idx = {m: i for i, m in enumerate(y_cols)}
        upper_bounds = {}
        for mat in clip_materials:
            if mat not in mat_to_idx:
                continue
            idx = mat_to_idx[mat]
            obs = y_train_mask[:, idx]
            if not np.any(obs):
                continue
            ub = np.quantile(y_train_raw[obs, idx], clip_upper_quantile)
            for arr, mask in [(y_train_raw, y_train_mask), (y_val_raw, y_val_mask),
                              (y_test_raw, y_test_mask)]:
                rows = mask[:, idx]
                arr[rows, idx] = np.minimum(arr[rows, idx], ub)
            upper_bounds[mat] = float(ub)
        clip_bounds = {"upper_quantile": clip_upper_quantile, "upper_bounds": upper_bounds}

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), ["Construction period"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["Typology", "Primary Code", "Hybrid Structure", "Country"]),
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
        groups_train=build_group_keys(X_train),
        groups_val  =build_group_keys(X_val),
        groups_test =build_group_keys(X_test),
        preprocessor=preprocessor,
        clip_bounds=clip_bounds,
        kept_rows=len(df),
        min_observed_targets=min_observed_targets,
    )


# ==========================================================
# Stage 1 — Material Occurrence Model  (XGBoost classifier)
# ==========================================================

class MaterialOccurrenceModel:
    """Classifier chain for joint material presence modeling.

    Trains one CalibratedClassifierCV(XGBClassifier, method='sigmoid') per
    material in descending binary-entropy order.  Each classifier conditions
    on all previously fitted materials, capturing co-occurrence dependencies
    that independent Bernoulli sampling misses.
    """

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
        self.chain_order_ = list(range(len(y_cols)))
        self.presence_calibrators_ = {}
        self.presence_group_calibrators_ = {}

    @staticmethod
    def _chain_X(X, ctx_cols):
        if not ctx_cols:
            return X
        return np.hstack([X, np.column_stack(ctx_cols)])

    def fit(self, X, y_presence):
        """Fit classifier chain in descending binary-entropy order.

        Each XGBClassifier is wrapped with CalibratedClassifierCV (Platt
        scaling) so that predict_proba() outputs calibrated probabilities
        directly.  Sigmoid is used for all materials: isotonic regression
        overfits on sparse probability bins (bins with < 15 samples between
        0.1 and 0.8), which inflates log-loss despite lower Brier score.
        """
        self.models_ = {}
        self.trivial_proba_ = {}
        p = y_presence.mean(axis=0).clip(1e-9, 1 - 1e-9)
        entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
        self.chain_order_ = list(np.argsort(-entropy))
        cal_method = "sigmoid"
        ctx = []
        for m in self.chain_order_:
            material = y_cols[m]
            obs = y_presence[:, m]
            p = float(obs.mean())
            if p == 0.0 or p == 1.0:
                self.trivial_proba_[material] = p
                self.models_[material] = None
            else:
                base = XGBClassifier(**self.xgb_params)
                clf = CalibratedClassifierCV(base, method=cal_method, cv=5)
                clf.fit(self._chain_X(X, ctx), obs.astype(int))
                self.models_[material] = clf
            ctx.append(obs.astype(np.float64))
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        proba = np.zeros((n, len(y_cols)), dtype=np.float64)
        ctx = []
        for m in self.chain_order_:
            material = y_cols[m]
            if self.models_.get(material) is None:
                p_m = np.full(n, self.trivial_proba_.get(material, 0.0))
            else:
                p_m = self.models_[material].predict_proba(
                    self._chain_X(X, ctx)
                )[:, 1]
            proba[:, m] = p_m
            ctx.append((p_m > 0.5).astype(np.float64))
        return proba

    def _predict_proba_order(self, X, perm, train_pos):
        n         = X.shape[0]
        proba     = np.zeros((n, len(y_cols)))
        ctx_built = {}

        for k in range(len(y_cols)):
            m        = perm[k]
            material = y_cols[m]
            t        = train_pos[m]

            ctx = [
                ctx_built.get(self.chain_order_[j], np.zeros(n))
                for j in range(t)
            ]

            if self.models_.get(material) is None:
                p_m = np.full(n, self.trivial_proba_.get(material, 0.0))
            else:
                p_m = self.models_[material].predict_proba(
                    self._chain_X(X, ctx)
                )[:, 1]

            proba[:, m]  = p_m
            ctx_built[m] = (p_m > 0.5).astype(np.float64)

        return proba

    def sample_presence(self, X, n_samples=1000, temperature=1.0, random_state=None,
                        n_chain_orders=4, primary_codes=None):
        """Draw coherent material combinations via randomized ancestral chain sampling."""
        _eps = 1e-9

        rng    = np.random.default_rng(random_state)
        n_rows = X.shape[0]
        M      = len(y_cols)
        out    = np.zeros((n_rows, n_samples, M), dtype=bool)

        train_pos = {m: pos for pos, m in enumerate(self.chain_order_)}

        for i in range(n_rows):
            Xi = X[i:i+1]

            p_avg = self.predict_proba(Xi)[0].copy()
            for _ in range(n_chain_orders - 1):
                perm  = rng.permutation(M)
                p_avg += self._predict_proba_order(Xi, perm, train_pos)[0]
            p_marginal = p_avg / n_chain_orders

            pres  = np.zeros((n_samples, M), dtype=bool)
            edges = np.linspace(0, n_samples, n_chain_orders + 1, dtype=int)

            for g in range(n_chain_orders):
                start, end = int(edges[g]), int(edges[g + 1])
                n_g   = end - start
                X_rep = np.tile(Xi, (n_g, 1))
                perm  = rng.permutation(M)
                pres_g    = np.zeros((n_g, M), dtype=bool)
                ctx_built = {}

                for k in range(M):
                    m        = perm[k]
                    material = y_cols[m]
                    t        = train_pos[m]

                    ctx = [
                        ctx_built.get(self.chain_order_[j], np.zeros(n_g))
                        for j in range(t)
                    ]

                    if self.models_.get(material) is None:
                        p_m = np.full(n_g, self.trivial_proba_.get(material, 0.0))
                    else:
                        p_chain = self.models_[material].predict_proba(
                            self._chain_X(X_rep, ctx)
                        )[:, 1]
                        if temperature != 1.0:
                            logit_p = np.log(p_chain.clip(_eps, 1 - _eps) /
                                             (1 - p_chain.clip(_eps, 1 - _eps)))
                            p_chain = 1.0 / (1.0 + np.exp(-logit_p / temperature))

                        p_base = np.full(n_g, float(p_marginal[m]))
                        p_m    = BLEND_LAMBDA * p_chain + (1.0 - BLEND_LAMBDA) * p_base

                    grp = primary_codes[i] if primary_codes is not None else None
                    cal = (self.presence_group_calibrators_.get((material, grp))
                           if grp is not None else None)
                    if cal is None:
                        cal = self.presence_calibrators_.get(material)
                    if cal is not None:
                        p_m = cal.predict(p_m).clip(0.0, 1.0)

                    p_m = np.clip(p_m, P_CLIP_LO, P_CLIP_HI)
                    z   = rng.random(n_g) < p_m
                    z   = z & (rng.random(n_g) >= DROPOUT_RATE)

                    pres_g[:, m] = z
                    ctx_built[m] = z.astype(np.float64)

                pres[start:end] = pres_g

            out[i] = pres

        return out


# ==========================================================
# _PerMaterialQuantileXGB — per-material quantile XGBoost helper
# ==========================================================

class _PerMaterialQuantileXGB:

    ALPHAS = [0.05, 0.50, 0.95]

    def __init__(self, xgb_params):
        self.xgb_params = xgb_params

    def fit(self, X, y_log):
        kw = {k: v for k, v in self.xgb_params.items() if k != "objective"}
        self.model_ = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=self.ALPHAS,
            **kw,
        )
        self.model_.fit(X, y_log)
        return self

    def predict_q_log(self, X):
        return self.model_.predict(X)

    def _sigma(self, pq):
        return np.maximum((pq[:, 2] - pq[:, 0]) / (2 * Z_QUANT_95), 1e-6)

    def predict_log_mean(self, X):
        return self.predict_q_log(X)[:, 1]

    def predict_quantiles(self, X, q_lo=0.05, q_hi=0.95):
        pq    = self.predict_q_log(X)
        mu    = pq[:, 1]
        sigma = self._sigma(pq)
        p_lo  = np.maximum(np.exp(mu + stats.norm.ppf(q_lo) * sigma) - LOG_EPS, 0.0)
        p50   = np.maximum(np.exp(mu) - LOG_EPS, 0.0)
        p_hi  = np.maximum(np.exp(mu + stats.norm.ppf(q_hi) * sigma) - LOG_EPS, 0.0)
        return p_lo, p50, p_hi

    def crps_gaussian(self, X, y_log):
        pq    = self.predict_q_log(X)
        mu    = pq[:, 1]
        sigma = self._sigma(pq)
        z = (y_log - mu) / sigma
        crps = sigma * (
            z * (2.0 * stats.norm.cdf(z) - 1.0)
            + 2.0 * stats.norm.pdf(z)
            - 1.0 / np.sqrt(np.pi)
        )
        return float(np.mean(crps))


# ==========================================================
# Stage 2 — Quantile XGBoost Intensity Model
# ==========================================================

class MaterialIntensityModel:

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

    def fit(self, X, y_raw, y_presence):
        self.models_ = {}
        for m, material in enumerate(y_cols):
            obs = y_presence[:, m]
            if obs.sum() < 2:
                self.models_[material] = None
                continue
            y_log = np.log(y_raw[obs, m] + LOG_EPS)
            qxgb = _PerMaterialQuantileXGB(xgb_params=self.xgb_params)
            qxgb.fit(X[obs], y_log)
            self.models_[material] = qxgb
        return self

    def predict_log(self, X):
        N = X.shape[0]
        mu_log = np.zeros((N, len(y_cols)), dtype=np.float64)
        for m, material in enumerate(y_cols):
            qxgb = self.models_.get(material)
            if qxgb is not None:
                mu_log[:, m] = qxgb.predict_log_mean(X)
        return mu_log

    def predict(self, X):
        return np.maximum(np.exp(self.predict_log(X)) - LOG_EPS, 0.0)

    def predict_intervals(self, X, alpha=0.10):
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
        result = {}
        N = X.shape[0]
        for material in y_cols:
            qxgb = self.models_.get(material)
            if qxgb is None:
                result[material] = {"p5": np.zeros(N), "p50": np.zeros(N), "p95": np.zeros(N)}
            else:
                p_lo, p50, p_hi = qxgb.predict_quantiles(X, q_lo=q_lo, q_hi=q_hi)
                result[material] = {"p5": p_lo, "p50": p50, "p95": p_hi}
        return result

    def evaluate_crps(self, X, y):
        if hasattr(y, "to_numpy"):
            y = y.to_numpy(dtype=np.float64)
        print(f"  {'Material':<12}  {'n_obs':>6}  {'Mean CRPS (log-space)':>22}")
        print("  " + "-" * 44)
        for m, material in enumerate(y_cols):
            qxgb = self.models_.get(material)
            if qxgb is None:
                print(f"  {material:<12}  {'—':>6}  {'no model':>22}")
                continue
            obs = (~np.isnan(y[:, m])) & (y[:, m] > 0)
            if obs.sum() < 2:
                print(f"  {material:<12}  {'—':>6}  {'too few rows':>22}")
                continue
            y_log = np.log(y[obs, m] + LOG_EPS)
            crps = qxgb.crps_gaussian(X[obs], y_log)
            print(f"  {material:<12}  {int(obs.sum()):>6}  {crps:>22.4f}")

    def evaluate_calibration(self, X, y, levels=None):
        if hasattr(y, "to_numpy"):
            y = y.to_numpy(dtype=np.float64)
        if levels is None:
            levels = np.linspace(0.10, 0.90, 9)
        levels = np.asarray(levels)

        results = {}
        level_hdr = "  ".join(f"{int(round(lv * 100)):3d}%" for lv in levels)
        print(f"  {'Material':<12}  {'n_obs':>6}    {level_hdr}")
        print("  " + "-" * (22 + 7 * len(levels)))

        for m, material in enumerate(y_cols):
            qxgb = self.models_.get(material)
            if qxgb is None:
                print(f"  {material:<12}  {'—':>6}    no model")
                continue
            obs = (~np.isnan(y[:, m])) & (y[:, m] > 0)
            if obs.sum() < 2:
                print(f"  {material:<12}  {'—':>6}    too few rows")
                continue
            y_obs = y[obs, m]
            X_obs = X[obs]
            emp = []
            for lv in levels:
                alpha = 1.0 - lv
                p_lo, _, p_hi = qxgb.predict_quantiles(X_obs, q_lo=alpha / 2.0, q_hi=1.0 - alpha / 2.0)
                emp.append(float(((y_obs >= p_lo) & (y_obs <= p_hi)).mean()))
            emp = np.array(emp)
            results[material] = emp
            emp_row = "  ".join(f"{v:.2f}" for v in emp)
            print(f"  {material:<12}  {int(obs.sum()):>6}    {emp_row}")

        return results


# ==========================================================
# Stage 3 — Joint Distribution Layer  (multivariate normal)
# ==========================================================

class JointDistributionModel:

    def __init__(self, group_cols=None, min_group_size=20, reg_eps=1e-4, cov_shrink=0.0):
        if group_cols is None:
            group_cols = GROUP_COLS
        self.group_cols = tuple(group_cols)
        self.min_group_size = min_group_size
        self.reg_eps = reg_eps
        self.cov_shrink = float(np.clip(cov_shrink, 0.0, 1.0))
        self.global_cov_ = None
        self.group_covs_ = {}

    def _pairwise_cov(self, residuals, presence):
        M = len(y_cols)
        cov = np.zeros((M, M))
        for m1 in range(M):
            for m2 in range(m1, M):
                both = presence[:, m1] & presence[:, m2]
                if both.sum() < 2:
                    continue
                r1 = residuals[both, m1]
                r2 = residuals[both, m2]
                if m1 == m2:
                    cov[m1, m1] = float(np.var(r1, ddof=1))
                else:
                    c = float(np.cov(r1, r2, ddof=1)[0, 1])
                    cov[m1, m2] = cov[m2, m1] = c
        return cov

    def _regularise_cov(self, cov, ref=None):
        if ref is None:
            ref = np.diag(np.diag(cov))
        shrunk = (1.0 - self.cov_shrink) * cov + self.cov_shrink * ref
        eye_M = np.eye(len(y_cols))
        return shrunk + eye_M * self.reg_eps

    def fit(self, X_proc, X_raw, y_raw, y_presence, intensity_model):
        usable = y_presence.sum(axis=1) >= 2
        if usable.sum() < len(y_cols):
            raise ValueError(f"Need >= {len(y_cols)} pairwise-usable rows; got {usable.sum()}.")

        mu_log = intensity_model.predict_log(X_proc)

        p_sub = y_presence[usable]
        y_sub = y_raw[usable].astype(np.float64)
        y_log = np.where(p_sub, np.log(np.where(p_sub, y_sub + LOG_EPS, 1.0)), np.nan)
        residuals = y_log - mu_log[usable]

        groups = build_group_keys(X_raw.loc[usable].reset_index(drop=True), self.group_cols)

        all_groups_raw = build_group_keys(X_raw.reset_index(drop=True), self.group_cols)
        uniq_raw = np.unique(all_groups_raw)
        uniq_use = np.unique(groups)

        print(f"\nJointDistributionModel.fit()  |  eligibility: ≥2 materials present")
        print(f"  usable rows: {usable.sum()} / {len(usable)}\n")

        print(f"  {'Group':<10}  {'Raw':>6}  {'Usable':>8}  {'Decision':>14}")
        print("  " + "-" * 46)
        for g in uniq_raw:
            raw_n = int((all_groups_raw == g).sum())
            use_n = int((groups == g).sum()) if g in uniq_use else 0
            decision = "RETAINED" if use_n >= self.min_group_size else f"DROPPED (<{self.min_group_size})"
            print(f"  {g:<10}  {raw_n:>6}  {use_n:>8}  {decision:>14}")

        print(f"\n  Usable rows per material pair:")
        print(f"  {'Pair':<26}  {'n_rows':>7}")
        print("  " + "-" * 36)
        for m1 in range(len(y_cols)):
            for m2 in range(m1, len(y_cols)):
                both = p_sub[:, m1] & p_sub[:, m2]
                print(f"  ({y_cols[m1]:<10}, {y_cols[m2]:<10})  {int(both.sum()):>7}")

        global_raw = self._pairwise_cov(residuals, p_sub)
        self.global_cov_ = self._regularise_cov(global_raw)

        self.group_covs_ = {}
        print(f"\n  Group covariance  (min_group_size={self.min_group_size}):")
        print(f"  {'Group':<10}  {'Rows':>6}  {'Raw rank':>9}  {'Decision':>10}")
        print("  " + "-" * 44)
        for g in uniq_use:
            g_mask = groups == g
            n      = int(g_mask.sum())
            g_res  = residuals[g_mask]
            g_pres = p_sub[g_mask]
            if n >= self.min_group_size:
                g_cov_raw = self._pairwise_cov(g_res, g_pres)
                self.group_covs_[g] = self._regularise_cov(g_cov_raw, ref=self.global_cov_)
                rank = int(np.linalg.matrix_rank(g_cov_raw))
                print(f"  {g:<10}  {n:>6}  {rank:>9}  {'RETAINED':>10}")
            else:
                print(f"  {g:<10}  {n:>6}  {'—':>9}  {'DROPPED':>10}")

        print(f"\n  Retained groups: {sorted(self.group_covs_.keys())}")
        return self

    def get_cov(self, group):
        return self.group_covs_.get(group, self.global_cov_)


# ==========================================================
# Two-Stage Conditional Probabilistic Model — wrapper + training
# ==========================================================

class TwoStageConditionalModel:

    def __init__(self, group_cols=GROUP_COLS, n_estimators=300, max_depth=4,
                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.0, reg_lambda=1.0,
                 min_group_size=20, reg_eps=1e-4, cov_shrink=0.0,
                 random_state=SEED):
        xgb_kw = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
            reg_lambda=reg_lambda, random_state=random_state,
        )
        self.stage1 = MaterialOccurrenceModel(**xgb_kw)
        self.stage2 = MaterialIntensityModel(**xgb_kw)
        self.joint  = JointDistributionModel(group_cols=group_cols,
                                              min_group_size=min_group_size,
                                              reg_eps=reg_eps,
                                              cov_shrink=0.4)

    def fit(self, X_proc, X_raw, y_raw, y_mask):
        y_presence = y_mask & (y_raw > 0)
        self.stage1.fit(X_proc, y_presence)
        self.stage2.fit(X_proc, y_raw, y_presence)
        self.joint.fit(X_proc, X_raw, y_raw, y_presence, self.stage2)

        self.intensity_bounds_ = []
        for m in range(len(y_cols)):
            obs = y_presence[:, m]
            if obs.sum() >= 2:
                vals = y_raw[obs, m]
                self.intensity_bounds_.append(
                    (float(np.percentile(vals, 1)), float(np.percentile(vals, 99)))
                )
            else:
                self.intensity_bounds_.append((0.0, np.inf))

        return self

    def predict(self, X_proc, groups, alpha=0.10):
        proba     = self.stage1.predict_proba(X_proc)
        intervals = self.stage2.predict_intervals(X_proc, alpha=alpha)
        for m, mat in enumerate(y_cols):
            intervals[mat]["p_presence"] = proba[:, m]
        return intervals

    @staticmethod
    def _marginal_resample(presence, p_target, rng):
        p_target = np.clip(p_target, P_CLIP_LO, P_CLIP_HI)
        n_rows, n_samples, M = presence.shape
        for i in range(n_rows):
            for m in range(M):
                freq  = float(presence[i, :, m].mean())
                delta = int(round((freq - float(p_target[i, m])) * n_samples))
                if delta > 0:
                    on_idx = np.where(presence[i, :, m])[0]
                    flip   = rng.choice(on_idx, size=min(delta, len(on_idx)), replace=False)
                    presence[i, flip, m] = False
                elif delta < 0:
                    off_idx = np.where(~presence[i, :, m])[0]
                    flip    = rng.choice(off_idx, size=min(-delta, len(off_idx)), replace=False)
                    presence[i, flip, m] = True
        return presence

    def sample_query(self, X_proc, groups, n_samples=1000, temperature=2.5, random_state=None):
        """Sample from the full joint predictive distribution."""
        rng    = np.random.default_rng(random_state)
        n_rows = X_proc.shape[0]
        M      = len(y_cols)

        all_presence = self.stage1.sample_presence(
            X_proc, n_samples=n_samples, temperature=temperature, random_state=rng,
            primary_codes=groups,
        )

        p_target = self.stage1.predict_proba(X_proc)
        self._marginal_resample(all_presence, p_target, rng)

        all_samples = np.zeros((n_rows, n_samples, M), dtype=np.float64)

        for i in range(n_rows):
            Xi    = X_proc[i : i + 1]
            Sigma = self.joint.get_cov(groups[i])
            Z     = all_presence[i]

            for s in range(n_samples):
                y_log    = np.zeros(M)
                sigma_sq = np.zeros(M)

                for m, material in enumerate(y_cols):
                    if not Z[s, m]:
                        continue
                    qxgb = self.stage2.models_.get(material)
                    if qxgb is None:
                        continue

                    pq_log = qxgb.predict_q_log(Xi)
                    mu     = float(pq_log[0, 1])
                    sigma  = max(float(pq_log[0, 2] - pq_log[0, 0]) / (2 * Z_QUANT_95), 1e-6)
                    y_log[m]    = rng.normal(mu, sigma)
                    sigma_sq[m] = sigma * sigma

                active = Z[s]
                if active.sum() >= 2:
                    idx      = np.where(active)[0]
                    Sig_sub  = Sigma[np.ix_(idx, idx)]
                    Sig_damp = CROSS_DAMP * Sig_sub
                    diag_pos = np.arange(len(idx))
                    Sig_damp[diag_pos, diag_pos] = Sig_sub[diag_pos, diag_pos]
                    eps_joint = rng.multivariate_normal(np.zeros(len(idx)), Sig_damp)
                    eps_joint *= RESIDUAL_SCALE
                    y_log[idx] += eps_joint

                for m in np.where(Z[s])[0]:
                    sigma_eff_sq = sigma_sq[m] + RESIDUAL_SCALE**2 * float(Sigma[m, m])
                    y_log[m] -= 0.5 * sigma_eff_sq

                y = np.maximum(np.exp(y_log) - LOG_EPS, 0.0)
                y *= Z[s]
                if self.intensity_bounds_ is not None:
                    for m in np.where(Z[s])[0]:
                        lo, hi = self.intensity_bounds_[m]
                        y[m] = np.clip(y[m], lo, hi)
                all_samples[i, s] = y

        return all_samples, all_presence

    def calibrate_sampling(
        self,
        X_val,
        y_val_presence,
        groups_val=None,
        n_samples=2000,
        temperature=2.5,
        random_state=SEED,
        min_group_size=15,
    ):
        """Post-hoc calibration of joint sampling thresholds."""
        all_pres = self.stage1.sample_presence(
            X_val,
            n_samples=n_samples,
            temperature=temperature,
            random_state=random_state,
        )

        sampled_freq = all_pres.mean(axis=1)

        for m, mat in enumerate(y_cols):
            x_cal = sampled_freq[:, m]
            y_cal = y_val_presence[:, m].astype(float)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x_cal, y_cal)
            self.stage1.presence_calibrators_[mat] = iso

        self.stage1.presence_group_calibrators_ = {}
        if groups_val is not None:
            groups_val = np.asarray(groups_val)
            for g in np.unique(groups_val):
                g_mask = groups_val == g
                if int(g_mask.sum()) < min_group_size:
                    continue
                for m, mat in enumerate(y_cols):
                    x_g = sampled_freq[g_mask, m]
                    y_g = y_val_presence[g_mask, m].astype(float)
                    if y_g.sum() < 2 or (1 - y_g).sum() < 2:
                        continue
                    iso_g = IsotonicRegression(out_of_bounds="clip")
                    iso_g.fit(x_g, y_g)
                    self.stage1.presence_group_calibrators_[(mat, g)] = iso_g

        return self

    def evaluate_sampling_realism(
        self, X_proc, groups,
        y_train_raw, y_train_mask, X_train_raw=None,
        n_samples=1000, random_state=None,
    ):
        samples, presence = self.sample_query(
            X_proc, groups, n_samples=n_samples, random_state=random_state
        )

        train_groups = (build_group_keys(X_train_raw)
                        if X_train_raw is not None else None)
        M   = len(y_cols)
        SEP = "=" * 60

        for i in range(X_proc.shape[0]):
            samp = samples[i]
            pres = presence[i]
            grp  = groups[i]

            print(f"\n{SEP}")
            print(f"Query row {i}  |  group: {grp}  |  n_samples={n_samples}")
            print(SEP)

            print("\nA. Presence frequency per material:")
            for m, mat in enumerate(y_cols):
                freq = float(pres[:, m].mean())
                bar  = "#" * int(round(freq * 20))
                print(f"   {mat:<12}  {freq:.3f}  |{bar:<20}|")

            print("\nB. Co-occurrence matrix  (fraction of samples):")
            print("             " + "  ".join(f"{mat[:5]:>5}" for mat in y_cols))
            for m1, mat1 in enumerate(y_cols):
                row = f"  {mat1:<12}"
                for m2 in range(M):
                    v = float((pres[:, m1] & pres[:, m2]).mean())
                    row += f"  {v:.2f}"
                print(row)

            print("\nC. Sampled intensity distribution  (kg/m², presence rows only):")
            print(f"   {'Material':<12}  {'n_pres':>6}  {'mean':>8}  "
                  f"{'median':>8}  {'p5':>8}  {'p95':>8}")
            print("   " + "-" * 58)
            for m, mat in enumerate(y_cols):
                vals = samp[pres[:, m], m]
                if len(vals) == 0:
                    print(f"   {mat:<12}  {'—':>6}  (never present in samples)")
                    continue
                print(
                    f"   {mat:<12}  {len(vals):>6}  {vals.mean():>8.1f}  "
                    f"{np.median(vals):>8.1f}  "
                    f"{np.percentile(vals,  5):>8.1f}  "
                    f"{np.percentile(vals, 95):>8.1f}"
                )

            combos, cnts = np.unique(pres.astype(np.int8), axis=0, return_counts=True)
            n_unique = len(combos)
            print(f"\nD. Diversity: {n_unique} unique material combinations"
                  f" / {n_samples} samples")
            print("   Top 5 combinations:")
            for idx in np.argsort(-cnts)[:5]:
                labels    = [y_cols[m] for m in range(M) if combos[idx, m]]
                label_str = ", ".join(labels) if labels else "(no materials)"
                print(f"     {label_str:<50}  {cnts[idx]:4d}  ({cnts[idx]/n_samples:.1%})")

            print("\nE. Comparison to training data:")
            if train_groups is not None:
                ref_mask = train_groups == grp
                n_ref    = int(ref_mask.sum())
                scope    = f"group='{grp}'"
            else:
                ref_mask = np.ones(y_train_raw.shape[0], dtype=bool)
                n_ref    = int(ref_mask.sum())
                scope    = "all training data"

            if n_ref == 0:
                ref_mask = np.ones(y_train_raw.shape[0], dtype=bool)
                n_ref    = int(ref_mask.sum())
                scope    = f"all training  (no rows for group='{grp}')"

            print(f"   Reference: {n_ref} buildings  ({scope})")
            y_ref = y_train_raw[ref_mask]
            m_ref = y_train_mask[ref_mask]

            print(
                f"   {'Material':<12}  {'Real pres%':>10}  {'Samp pres%':>10}  "
                f"{'Real mean':>10}  {'Samp mean':>10}  {'Status':>8}"
            )
            print("   " + "-" * 70)
            for m, mat in enumerate(y_cols):
                real_rows = m_ref[:, m] & (y_ref[:, m] > 0)
                samp_rows = pres[:, m]
                real_pf   = float(real_rows.mean())
                samp_pf   = float(samp_rows.mean())
                real_mu   = (float(y_ref[real_rows, m].mean())
                             if real_rows.sum() > 0 else np.nan)
                samp_mu   = (float(samp[samp_rows, m].mean())
                             if samp_rows.sum() > 0 else np.nan)

                pres_ok = abs(real_pf - samp_pf) < 0.15
                mean_ok = (np.isnan(real_mu) or np.isnan(samp_mu)
                           or abs(real_mu - samp_mu) / (real_mu + 1e-3) < 0.50)
                status  = "OK" if (pres_ok and mean_ok) else "CHECK"

                rm_s = f"{real_mu:>10.1f}" if not np.isnan(real_mu) else f"{'—':>10}"
                sm_s = f"{samp_mu:>10.1f}" if not np.isnan(samp_mu) else f"{'—':>10}"
                print(
                    f"   {mat:<12}  {real_pf:>10.3f}  {samp_pf:>10.3f}  "
                    f"{rm_s}  {sm_s}  {status:>8}"
                )

    def evaluate_samples(
        self, samples, presence, y_ref, y_ref_mask, groups_ref=None, query_groups=None,
    ):
        M = len(y_cols)
        results = {}

        for m, mat in enumerate(y_cols):
            samp_freq = float(presence[:, :, m].mean())
            true_freq = float(y_ref_mask[:, m].mean())
            pres_err  = abs(samp_freq - true_freq)

            samp_vals = samples[:, :, m][presence[:, :, m]]
            ref_rows  = y_ref_mask[:, m]
            ref_vals  = y_ref[ref_rows, m]

            if len(samp_vals) < 2 or len(ref_vals) < 2:
                results[mat] = dict(pres_freq_err=pres_err, kl_div=np.nan, wasserstein1=np.nan)
                continue

            samp_log = np.log(samp_vals + LOG_EPS)
            ref_log  = np.log(ref_vals  + LOG_EPS)

            w1 = float(wasserstein_distance(ref_log, samp_log))

            edges      = np.linspace(
                min(ref_log.min(), samp_log.min()),
                max(ref_log.max(), samp_log.max()),
                31,
            )
            p_ref,  _  = np.histogram(ref_log,  bins=edges, density=True)
            p_samp, _  = np.histogram(samp_log, bins=edges, density=True)
            _eps2      = 1e-10
            p_ref      = p_ref  + _eps2;  p_ref  /= p_ref.sum()
            p_samp     = p_samp + _eps2;  p_samp /= p_samp.sum()
            kl         = float(np.sum(p_ref * np.log(p_ref / p_samp)))

            results[mat] = dict(pres_freq_err=pres_err, kl_div=kl, wasserstein1=w1)

        return results

    def print_sample_metrics(
        self, samples, presence, y_ref, y_ref_mask, groups_ref=None, query_groups=None,
    ):
        metrics = self.evaluate_samples(
            samples, presence, y_ref, y_ref_mask, groups_ref, query_groups,
        )
        print(f"\n{'Material':<12}  {'Pres |err|':>10}  {'KL div':>8}  {'Wass-1':>8}")
        print("-" * 44)
        for mat in y_cols:
            r  = metrics[mat]
            kl = f"{r['kl_div']:>8.4f}" if not np.isnan(r["kl_div"]) else f"{'—':>8}"
            w1 = f"{r['wasserstein1']:>8.4f}" if not np.isnan(r["wasserstein1"]) else f"{'—':>8}"
            print(f"{mat:<12}  {r['pres_freq_err']:>10.4f}  {kl}  {w1}")


# ==========================================================
# Hyperparameter tuning helpers
# ==========================================================

def compute_naive_maes(y_train_raw, y_train_mask):
    naive_maes = {}
    for m, mat in enumerate(y_cols):
        obs = y_train_mask[:, m] & (y_train_raw[:, m] > 0)
        if obs.sum() > 0:
            vals = y_train_raw[obs, m]
            naive_maes[mat] = float(np.mean(np.abs(vals - vals.mean())))
    return naive_maes


def compute_mase(pred_dict, y_df, naive_maes, eps=1e-8):
    mases = []
    for mat in y_cols:
        y_true = y_df[mat].to_numpy(dtype=float)
        obs = (~np.isnan(y_true)) & (y_true > 0)
        if np.any(obs):
            y_pred = pred_dict[mat]["p50"][obs]
            mae = np.mean(np.abs(y_pred - y_true[obs]))
            mases.append(mae / (naive_maes.get(mat, 1.0) + eps))
    return float(np.mean(mases)) if mases else np.inf


# ==========================================================
# Diagnostic helpers
# ==========================================================

def _stage1_reliability(stage1, X, y_presence, n_bins=10):
    proba = stage1.predict_proba(X)
    results = {}
    print(f"  {'Material':<12}  {'Brier':>8}  {'Mean pred':>10}  {'Obs freq':>10}  {'Bias':>8}")
    print("  " + "-" * 62)
    for m, mat in enumerate(y_cols):
        y_true  = y_presence[:, m].astype(float)
        p_pred  = proba[:, m]
        brier   = float(np.mean((p_pred - y_true) ** 2))
        pred_mu = float(p_pred.mean())
        obs_f   = float(y_true.mean())
        bins    = np.linspace(0.0, 1.0, n_bins + 1)
        mp_bins, fp_bins = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p_pred >= lo) & (p_pred < hi)
            if mask.sum() > 0:
                mp_bins.append(float(p_pred[mask].mean()))
                fp_bins.append(float(y_true[mask].mean()))
        print(f"  {mat:<12}  {brier:>8.4f}  {pred_mu:>9.3f}     {obs_f:>9.3f}  {pred_mu - obs_f:>+8.3f}")
        results[mat] = dict(
            mean_pred_bins=np.array(mp_bins),
            frac_pos_bins=np.array(fp_bins),
            brier=brier,
            pred_mean=pred_mu,
            obs_freq=obs_f,
        )
    return results


def show_quantile_spread(model, X, y_raw):
    print(f"{'Material':<12}  {'n_obs':>6}  {'Mean p50':>9}  "
          f"{'Mean IW':>9}  {'Data rng':>10}  {'Mean σ_log':>10}")
    print("-" * 67)
    for m, mat in enumerate(y_cols):
        qxgb = model.stage2.models_.get(mat)
        if qxgb is None:
            print(f"{mat:<12}  {'—':>6}  no model")
            continue
        obs = (~np.isnan(y_raw[:, m])) & (y_raw[:, m] > 0)
        if obs.sum() < 2:
            print(f"{mat:<12}  {'—':>6}  too few rows")
            continue
        pq        = qxgb.predict_q_log(X[obs])
        p05       = np.maximum(np.exp(pq[:, 0]) - 1e-6, 0.0)
        p50       = np.maximum(np.exp(pq[:, 1]) - 1e-6, 0.0)
        p95       = np.maximum(np.exp(pq[:, 2]) - 1e-6, 0.0)
        iw        = float((p95 - p05).mean())
        drange    = float(y_raw[obs, m].max() - y_raw[obs, m].min())
        sigma_log = float(((pq[:, 2] - pq[:, 0]) / (2 * 1.6449)).mean())
        print(f"{mat:<12}  {int(obs.sum()):>6}  {p50.mean():>9.1f}  "
              f"{iw:>9.1f}  {drange:>10.1f}  {sigma_log:>10.4f}")


def show_point_accuracy(model, X, y_raw):
    print(f"{'Material':<12}  {'n_obs':>6}  {'MAE':>9}  {'RMSE':>9}  {'R²':>7}")
    print("-" * 52)
    for m, mat in enumerate(y_cols):
        qxgb = model.stage2.models_.get(mat)
        if qxgb is None:
            print(f"{mat:<12}  {'—':>6}  no model")
            continue
        obs = (~np.isnan(y_raw[:, m])) & (y_raw[:, m] > 0)
        if obs.sum() < 2:
            print(f"{mat:<12}  {'—':>6}  too few rows")
            continue
        y_obs  = y_raw[obs, m]
        pq     = qxgb.predict_q_log(X[obs])
        p50    = np.maximum(np.exp(pq[:, 1]) - 1e-6, 0.0)
        mae    = float(np.mean(np.abs(p50 - y_obs)))
        rmse   = float(np.sqrt(np.mean((p50 - y_obs) ** 2)))
        ss_res = float(np.sum((y_obs - p50) ** 2))
        ss_tot = float(np.sum((y_obs - y_obs.mean()) ** 2))
        r2     = 1.0 - ss_res / (ss_tot + 1e-9)
        print(f"{mat:<12}  {int(obs.sum()):>6}  {mae:>9.2f}  {rmse:>9.2f}  {r2:>7.3f}")


def _gaussian_kl(mu_p, cov_p, mu_q, cov_q, eps=1e-6):
    d = mu_p.shape[0]
    cov_p_r = cov_p + np.eye(d) * eps
    cov_q_r = cov_q + np.eye(d) * eps
    sp, ldp = np.linalg.slogdet(cov_p_r)
    sq, ldq = np.linalg.slogdet(cov_q_r)
    if sp <= 0 or sq <= 0:
        return np.nan
    inv_q = np.linalg.inv(cov_q_r)
    return 0.5 * (np.trace(inv_q @ cov_p_r)
                  + (mu_q - mu_p) @ inv_q @ (mu_q - mu_p)
                  - d + (ldq - ldp))


# ==========================================================
# Main workflow
# ==========================================================

if __name__ == "__main__":
    # ── Data preparation ──────────────────────────────────────────────────────
    set_global_seed(SEED)
    data = prepare_data()

    print("Data preparation complete.")
    print(f"X_train: {data['X_train_proc'].shape}  y_train: {data['y_train_raw'].shape}")
    print(f"X_val:   {data['X_val_proc'].shape}  y_val:   {data['y_val_raw'].shape}")
    print(f"X_test:  {data['X_test_proc'].shape}  y_test:  {data['y_test_raw'].shape}")
    print(f"Rows kept: {data['kept_rows']}  (min observed targets: {data['min_observed_targets']})")

    for split in ["train", "val", "test"]:
        mask = data[f"y_{split}_mask"]
        obs  = mask.sum(axis=1)
        print(f"  {split}: obs/row  min={obs.min()}  mean={obs.mean():.2f}  max={obs.max()}")

    # ── Initial model fit ─────────────────────────────────────────────────────
    set_global_seed(SEED)
    model = TwoStageConditionalModel(random_state=SEED)
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(data["X_train_proc"], data["X_train_raw"], data["y_train_raw"], data["y_train_mask"])
    print("Model fitted  |  groups retained:", sorted(model.joint.group_covs_.keys()))

    y_val_presence = data["y_val_mask"] & (data["y_val_raw"] > 0)
    model.calibrate_sampling(
        data["X_val_proc"], y_val_presence,
        groups_val=data["groups_val"],
        n_samples=2000, temperature=2.5, random_state=SEED,
    )
    print("Joint sampling calibration fitted.")

    # ── Stage 1 reliability diagnostics ──────────────────────────────────────
    y_val_presence  = data["y_val_mask"]  & (data["y_val_raw"]  > 0)
    y_test_presence = data["y_test_mask"] & (data["y_test_raw"] > 0)

    print("=" * 60)
    print("Stage 1 reliability — TEST SET  (Platt-calibrated)")
    print("=" * 60)
    cal_result = _stage1_reliability(model.stage1, data["X_test_proc"], y_test_presence)

    fig, axes = plt.subplots(1, len(y_cols), figsize=(16, 4), sharey=True)
    for ax, mat in zip(axes, y_cols):
        res = cal_result.get(mat, {})
        mp  = res.get("mean_pred_bins")
        fp  = res.get("frac_pos_bins")
        if mp is not None and len(mp):
            ax.plot(mp, fp, "o-", color="steelblue", lw=1.5, ms=5, label="Platt sigmoid")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Perfect")
        brier = res.get("brier", float("nan"))
        ax.set_title(f"{mat}\nBrier={brier:.4f}", fontsize=10)
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Fraction positive")
    fig.suptitle("Stage 1 Reliability Curves — Test Set (Platt-calibrated)", fontsize=11)
    plt.tight_layout()
    plt.show()

    # ── Stage 2 quantile spread diagnostics ──────────────────────────────────
    print("=" * 67)
    print("Stage 2 Quantile Spread — VALIDATION SET")
    print("=" * 67)
    show_quantile_spread(model, data["X_val_proc"], data["y_val_raw"])

    print()
    print("=" * 67)
    print("Stage 2 Quantile Spread — TEST SET")
    print("=" * 67)
    show_quantile_spread(model, data["X_test_proc"], data["y_test_raw"])

    # ── Stage 2 point estimate accuracy ──────────────────────────────────────
    print("=" * 52)
    print("Point Estimate Accuracy — VALIDATION SET")
    print("=" * 52)
    show_point_accuracy(model, data["X_val_proc"], data["y_val_raw"])

    print()
    print("=" * 52)
    print("Point Estimate Accuracy — TEST SET")
    print("=" * 52)
    show_point_accuracy(model, data["X_test_proc"], data["y_test_raw"])

    # ── CRPS ──────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("CRPS evaluation — VALIDATION SET")
    print("=" * 60)
    model.stage2.evaluate_crps(data["X_val_proc"], data["y_val_raw"])

    print()
    print("=" * 60)
    print("CRPS evaluation — TEST SET")
    print("=" * 60)
    model.stage2.evaluate_crps(data["X_test_proc"], data["y_test_raw"])

    # ── Calibration coverage ──────────────────────────────────────────────────
    levels = np.linspace(0.10, 0.90, 9)

    print("Calibration — VALIDATION SET")
    cal_val  = model.stage2.evaluate_calibration(data["X_val_proc"],  data["y_val_raw"],  levels=levels)
    print()
    print("Calibration — TEST SET")
    cal_test = model.stage2.evaluate_calibration(data["X_test_proc"], data["y_test_raw"], levels=levels)

    fig, axes = plt.subplots(1, len(y_cols), figsize=(16, 4), sharey=True)
    for ax, material in zip(axes, y_cols):
        emp_v = cal_val.get(material)
        emp_t = cal_test.get(material)
        if emp_v is not None:
            ax.plot(levels, emp_v, "o-",  color="steelblue", lw=1.5, ms=5, label="Val")
        if emp_t is not None:
            ax.plot(levels, emp_t, "s--", color="coral",     lw=1.5, ms=5, label="Test")
        ax.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.4, label="Perfect")
        ax.set_title(material, fontsize=11)
        ax.set_xlabel("Nominal coverage")
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(levels)
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.legend(fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Empirical coverage")
    fig.suptitle(
        "Calibration: Nominal vs Empirical Coverage\n(presence rows only, quantile XGBoost log-space)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()

    # ── Sampling realism diagnostics ──────────────────────────────────────────
    n_query   = 3
    X_query   = data["X_val_proc"][:n_query]
    grp_query = data["groups_val"][:n_query]

    model.evaluate_sampling_realism(
        X_query, grp_query,
        y_train_raw  = data["y_train_raw"],
        y_train_mask = data["y_train_mask"],
        X_train_raw  = data["X_train_raw"],
        n_samples    = 1000,
        random_state = SEED,
    )

    # ── Optuna hyperparameter tuning ──────────────────────────────────────────
    naive_maes = compute_naive_maes(data["y_train_raw"], data["y_train_mask"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        n_estimators     = trial.suggest_int("n_estimators", 100, 600)
        max_depth        = trial.suggest_int("max_depth", 3, 7)
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample        = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
        cov_shrink       = trial.suggest_float("cov_shrink", 0.0, 1.0)
        min_group_size   = trial.suggest_int("min_group_size", 10, 40)
        reg_eps          = trial.suggest_float("reg_eps", 1e-6, 1e-2, log=True)

        tuned_model = TwoStageConditionalModel(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
            reg_lambda=reg_lambda, cov_shrink=cov_shrink,
            min_group_size=min_group_size, reg_eps=reg_eps,
            random_state=SEED,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            tuned_model.fit(
                data["X_train_proc"], data["X_train_raw"],
                data["y_train_raw"],  data["y_train_mask"],
            )
        preds_val = tuned_model.predict(data["X_val_proc"], data["groups_val"], alpha=0.10)
        return compute_mase(preds_val, data["y_val_df"], naive_maes)

    set_global_seed(SEED)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))

    def _cb(study, trial):
        if trial.number % 10 == 9 or trial.number == 0:
            print(f"  trial {trial.number+1:3d}/50  best MASE={study.best_value:.4f}")

    study.optimize(objective, n_trials=50, show_progress_bar=False, callbacks=[_cb])

    print("Best validation MASE:", round(study.best_value, 6))
    print("Best params:", study.best_params)

    best = study.best_params
    model = TwoStageConditionalModel(
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
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(
            data["X_train_proc"], data["X_train_raw"],
            data["y_train_raw"],  data["y_train_mask"],
        )
    print("Tuned model fitted  |  groups retained:", sorted(model.joint.group_covs_.keys()))

    y_val_presence = data["y_val_mask"] & (data["y_val_raw"] > 0)
    model.calibrate_sampling(
        data["X_val_proc"], y_val_presence,
        groups_val=data["groups_val"],
        n_samples=2000, temperature=2.5, random_state=SEED,
    )
    print("Joint sampling calibration re-fitted on tuned model.")

    # ── Prediction on test set ────────────────────────────────────────────────
    pred_test = model.predict(data["X_test_proc"], data["groups_test"], alpha=0.10)

    print("Sample predictions (first 3 test buildings):\n")
    for mat in y_cols:
        p = pred_test[mat]
        print(
            f"{mat:<10s}  "
            f"p_presence={np.round(p['p_presence'][:3], 2)}  "
            f"p50={np.round(p['p50'][:3], 1)}  "
            f"90%PI=[{np.round(p['p5'][:3], 1)}, {np.round(p['p95'][:3], 1)}]"
        )

    # ── Stage 2 log-residual normality check ──────────────────────────────────
    y_test_presence = data["y_test_mask"] & (data["y_test_raw"] > 0)
    X_test  = data["X_test_proc"]
    y_test  = data["y_test_raw"]

    residuals_dict = {}
    sigma_table    = []

    for m, mat in enumerate(y_cols):
        qxgb = model.stage2.models_.get(mat)
        obs  = y_test_presence[:, m]
        if qxgb is None or obs.sum() < 5:
            continue

        y_log_true = np.log(y_test[obs, m] + LOG_EPS)
        mu_log     = qxgb.predict_log_mean(X_test[obs])
        resid      = y_log_true - mu_log
        residuals_dict[mat] = resid

        pq         = qxgb.predict_q_log(X_test[obs])
        sigma_rec  = float(np.mean((pq[:, 2] - pq[:, 0]) / (2 * Z_QUANT_95)))
        sigma_emp  = float(np.std(resid, ddof=1))

        sw_stat, sw_p = shapiro(resid) if len(resid) <= 5000 else (float("nan"), float("nan"))
        verdict = "✓ normal" if (sw_p > 0.05 or np.isnan(sw_p)) else "✗ non-normal"

        sigma_table.append(dict(
            material=mat, n=int(obs.sum()),
            sigma_emp=sigma_emp, sigma_rec=sigma_rec,
            ratio=sigma_rec / (sigma_emp + 1e-9),
            sw_stat=sw_stat, sw_p=sw_p, verdict=verdict,
            mean_resid=float(np.mean(resid)),
        ))

    print(f"{'Material':<12}  {'n':>5}  {'mean_resid':>11}  "
          f"{'sigma_empirical':>15}  {'sigma_recovered':>15}  {'ratio':>6}  "
          f"{'Shapiro-W':>10}  {'p-val':>8}  {'Verdict'}")
    print("-" * 100)
    for r in sigma_table:
        sw  = f"{r['sw_stat']:.4f}" if not np.isnan(r['sw_stat']) else "   n/a"
        swp = f"{r['sw_p']:.4f}"   if not np.isnan(r['sw_p'])    else "   n/a"
        print(f"  {r['material']:<10}  {r['n']:>5}  {r['mean_resid']:>11.4f}  "
              f"{r['sigma_emp']:>15.4f}  {r['sigma_rec']:>15.4f}  "
              f"{r['ratio']:>6.3f}  {sw:>10}  {swp:>8}  {r['verdict']}")

    print()
    print("ratio = sigma_recovered / sigma_empirical; values near 1.0 confirm the Gaussian assumption.")

    n_mats = len(residuals_dict)
    fig, axes = plt.subplots(2, n_mats, figsize=(4 * n_mats, 8))

    for col, (mat, resid) in enumerate(residuals_dict.items()):
        resid_std = (resid - resid.mean()) / (resid.std(ddof=1) + 1e-9)

        ax_qq = axes[0, col]
        (osm, osr), (slope, intercept, _) = probplot(resid_std, dist="norm")
        ax_qq.scatter(osm, osr, s=6, alpha=0.4, color="steelblue", rasterized=True)
        lo, hi = osm.min(), osm.max()
        ax_qq.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept],
                   "r-", lw=1.2, label=f"fit  slope={slope:.2f}")
        ax_qq.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="perfect N(0,1)")
        ax_qq.set_title(f"{mat}\nQQ plot (standardised)", fontsize=10)
        ax_qq.set_xlabel("Theoretical quantiles")
        ax_qq.set_ylabel("Sample quantiles")
        ax_qq.legend(fontsize=7)

        ax_h = axes[1, col]
        ax_h.hist(resid, bins=30, density=True, alpha=0.55,
                  color="steelblue", edgecolor="white", lw=0.4)
        mu_r, sd_r = resid.mean(), resid.std(ddof=1)
        x_fit = np.linspace(resid.min(), resid.max(), 200)
        ax_h.plot(x_fit, _norm.pdf(x_fit, mu_r, sd_r),
                  "r-", lw=1.8, label=f"N({mu_r:.2f}, {sd_r:.2f}²)")
        r = next(d for d in sigma_table if d["material"] == mat)
        ax_h.set_title(
            f"{mat}\nσ_emp={r['sigma_emp']:.3f}  σ_rec={r['sigma_rec']:.3f}",
            fontsize=10,
        )
        ax_h.set_xlabel("Log-space residual")
        ax_h.set_ylabel("Density")
        ax_h.legend(fontsize=7)

    fig.suptitle(
        "Stage 2 Log-Residual Normality — Test Set\n"
        "Validates Gaussian σ recovery: σ = (p95_log − p05_log) / (2 × 1.6449)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig("stage2_log_residual_normality.pdf", bbox_inches="tight", dpi=150)
    plt.show()
    print("Saved: stage2_log_residual_normality.pdf")

    # ── Visualisation — predicted median + 90% PI vs. true values ────────────
    n_plot = 50
    pred_plot = model.predict(data["X_test_proc"][:n_plot], data["groups_test"][:n_plot], alpha=0.10)

    fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=True)
    x_idx = np.arange(n_plot)

    for i, mat in enumerate(y_cols):
        ax = axes[i]
        y_true = data["y_test_df"][mat].iloc[:n_plot].to_numpy(dtype=float)
        observed = ~np.isnan(y_true)
        y_true_plot = y_true.copy()
        y_true_plot[~observed] = np.nan

        ax.fill_between(x_idx, pred_plot[mat]["p5"], pred_plot[mat]["p95"],
                        alpha=0.25, color="steelblue", label="90 % PI")
        ax.plot(x_idx, pred_plot[mat]["p50"], "bo", ms=4, label="Predicted median (p50)")
        ax.plot(x_idx, y_true_plot, "r*", ms=6, label="True value")
        ax.set_title(mat)
        ax.set_ylabel("MI (kg/m²)")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Test building index")
    plt.suptitle("Two-Stage Conditional Model — Test Set (first 50)", y=1.01)
    plt.tight_layout()
    plt.show()

    # ── Joint distribution evaluation ─────────────────────────────────────────
    cc = (~np.isnan(data["y_test_raw"]) & (data["y_test_raw"] > 0)).all(axis=1)
    y_tc = data["y_test_raw"][cc]
    y_pc = model.stage2.predict(data["X_test_proc"][cc])

    mu_t, mu_p   = y_tc.mean(0), y_pc.mean(0)
    std_t, std_p = y_tc.std(0, ddof=1), y_pc.std(0, ddof=1)
    cov_t  = np.cov(y_tc, rowvar=False)
    cov_p  = np.cov(y_pc, rowvar=False)
    corr_t = np.corrcoef(y_tc, rowvar=False)
    corr_p = np.corrcoef(y_pc, rowvar=False)

    print(f"Presence-complete test rows: {cc.sum()}")
    stats_summary = {
        "mean_vector_mae":  float(np.mean(np.abs(mu_t - mu_p))),
        "cov_frobenius":    float(np.linalg.norm(cov_t - cov_p, ord="fro")),
        "corr_frobenius":   float(np.linalg.norm(corr_t - corr_p, ord="fro")),
        "kl_true_to_pred":  float(_gaussian_kl(mu_t, cov_t, mu_p, cov_p)),
        "kl_pred_to_true":  float(_gaussian_kl(mu_p, cov_p, mu_t, cov_t)),
    }
    for k, v in stats_summary.items():
        print(f"  {k}: {v:.5g}")

    per_material = pd.DataFrame({
        "material": y_cols,
        "true_mean": mu_t, "pred_mean": mu_p, "abs_mean_diff": np.abs(mu_t - mu_p),
        "true_std":  std_t, "pred_std":  std_p, "abs_std_diff":  np.abs(std_t - std_p),
    })
    print(per_material.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, corr, title in zip(axes, [corr_t, corr_p],
                                ["True Correlation (Presence-Complete Cases)",
                                 "Predicted Median Correlation"]):
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                    xticklabels=y_cols, yticklabels=y_cols, ax=ax)
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

    # ── Save artefacts ────────────────────────────────────────────────────────
    try:
        import two_stage_model as _tsm
        model.__class__        = _tsm.TwoStageConditionalModel
        model.stage1.__class__ = _tsm.MaterialOccurrenceModel
        model.stage2.__class__ = _tsm.MaterialIntensityModel
        model.joint.__class__  = _tsm.JointDistributionModel
        for mat in y_cols:
            qxgb = model.stage2.models_.get(mat)
            if qxgb is not None:
                qxgb.__class__ = _tsm._PerMaterialQuantileXGB
    except ImportError:
        pass

    joblib.dump(data["preprocessor"], "preprocessor.joblib")
    joblib.dump(model, "model.joblib")

    model_info = {
        "model_type":  "TwoStageConditionalModel",
        "stage1":      "XGBClassifier per-material (binary:logistic, Platt-calibrated via CalibratedClassifierCV)",
        "stage2":      "XGBRegressor per-material quantile regression (reg:quantileerror, quantiles=[0.05,0.50,0.95], log-space)",
        "joint_layer": "MultivariateNormal with group-specific covariance (residual inspection only)",
        "group_cols":  list(GROUP_COLS),
        "y_cols":      y_cols,
        "X_cols":      X_cols,
        "stage1_xgb_params": model.stage1.xgb_params,
        "stage2_xgb_params": model.stage2.xgb_params,
    }
    with open("model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    print("Saved: preprocessor.joblib,  model.joblib,  model_info.json")

    # ── Validation 1 — coverage, interval width, median MAE ──────────────────
    coverage_results = []
    for mat in y_cols:
        y_true = data["y_test_df"][mat].to_numpy(dtype=float)
        obs    = (~np.isnan(y_true)) & (y_true > 0)
        y_true = y_true[obs]
        q05 = pred_test[mat]["p5"][obs]
        q50 = pred_test[mat]["p50"][obs]
        q95 = pred_test[mat]["p95"][obs]
        widths = q95 - q05
        coverage_results.append({
            "material":     mat,
            "coverage_90":  round(float(np.mean((y_true >= q05) & (y_true <= q95))), 4),
            "mean_width":   round(float(widths.mean()), 2),
            "median_width": round(float(np.median(widths)), 2),
            "p50_mae":      round(mean_absolute_error(y_true, q50), 4),
        })

    coverage_df = pd.DataFrame(coverage_results)
    print("===== Coverage / Width / p50 MAE  (presence rows) =====")
    print(coverage_df.to_string(index=False))

    # ── Paper validation — Task 1: stronger baseline comparison ───────────────
    stronger_baseline_rows = []

    for m, mat in enumerate(y_cols):
        tr_pres = data["y_train_mask"][:, m] & (data["y_train_raw"][:, m] > 0)
        te_pres = data["y_test_mask"][:,  m] & (data["y_test_raw"][:,  m] > 0)

        if tr_pres.sum() < 10 or te_pres.sum() < 3:
            continue

        X_tr = data["X_train_proc"][tr_pres]
        y_tr = data["y_train_raw"][tr_pres, m]
        X_te = data["X_test_proc"][te_pres]
        y_te = data["y_test_raw"][te_pres, m]

        median_val = float(np.median(y_tr))
        mae_median = float(np.mean(np.abs(y_te - median_val)))

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, y_tr)
        mae_ridge = float(np.mean(np.abs(y_te - ridge.predict(X_te))))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=SEED, n_jobs=-1)
            rf.fit(X_tr, y_tr)
        mae_rf = float(np.mean(np.abs(y_te - rf.predict(X_te))))

        qxgb = model.stage2.models_.get(mat)
        if qxgb is not None:
            p50_log = qxgb.predict_q_log(X_te)[:, 1]
            mae_model = float(np.mean(np.abs(y_te - (np.exp(p50_log) - LOG_EPS))))
        else:
            mae_model = float("nan")

        stronger_baseline_rows.append(dict(
            Material        = mat,
            n_test          = int(te_pres.sum()),
            MAE_Median      = round(mae_median, 2),
            MAE_Ridge       = round(mae_ridge,  2),
            MAE_RF          = round(mae_rf,     2),
            MAE_Stage2_p50  = round(mae_model,  2),
            vs_Median_pct   = round((mae_median - mae_model) / mae_median * 100, 1),
            vs_RF_pct       = round((mae_rf     - mae_model) / mae_rf     * 100, 1),
        ))

    df_stronger = pd.DataFrame(stronger_baseline_rows)
    print("=" * 80)
    print("STRONGER BASELINE — Stage 2 MAE on test presence rows")
    print("=" * 80)
    print(df_stronger.to_string(index=False))
    print()
    print("Columns: MAE in kg/m².  vs_X_pct = % improvement of Stage 2 over X.")
    print("Ridge: L2-regularised linear model (alpha=1.0), trained on presence rows.")
    print("RF:    RandomForestRegressor(n_estimators=300, max_depth=8).")

    # ── Paper validation — Task 2: structural prior sensitivity ───────────────
    print("STRUCTURAL_PRIOR removed — isotonic calibration subsumes the prior.")
    print("See sensitivity analysis output in prior cell run for evidence.")

    # ── Paper validation — Task 3: Stage 3 normality — QQ plots ──────────────
    mu_log_train = model.stage2.predict_log(data["X_train_proc"])

    fig, axes = plt.subplots(1, len(y_cols), figsize=(4 * len(y_cols), 4), sharey=False)
    fig.suptitle(
        "QQ Plots — Stage 2 Standardised Log-Residuals vs Normal\n"
        "(Validates Stage 3 multivariate-normal joint-residual assumption)",
        fontsize=11,
    )

    print(f"{'Material':<12}  {'n_obs':>6}  {'Shapiro-W':>10}  {'p-value':>10}  "
          f"{'KS stat':>8}  {'KS p':>8}  {'Verdict':>10}")
    print("-" * 72)

    for ax, (m, mat) in zip(axes, enumerate(y_cols)):
        obs = data["y_train_mask"][:, m] & (data["y_train_raw"][:, m] > 0)
        if obs.sum() < 8:
            ax.set_title(mat)
            ax.text(0.5, 0.5, "insufficient data", ha="center", transform=ax.transAxes)
            print(f"  {mat:<12}  {'—':>6}")
            continue

        y_log   = np.log(data["y_train_raw"][obs, m] + LOG_EPS)
        mu      = mu_log_train[obs, m]
        sigma   = model.stage2.models_[mat]._sigma(
                      model.stage2.models_[mat].predict_q_log(data["X_train_proc"][obs])
                  )
        resid   = (y_log - mu) / np.maximum(sigma, 1e-8)

        (osm, osr), (slope, intercept, r) = probplot(resid, dist="norm", plot=None)
        ax.plot(osm, osr,  "o", ms=3, alpha=0.4, color="#2b7bba", label="residuals")
        ax.plot(osm, slope * osm + intercept, "r--", lw=1.5, label="normal ref.")
        ax.set_title(f"{mat}  (R²={r**2:.3f})", fontsize=10)
        ax.set_xlabel("Theoretical quantiles")
        if m == 0:
            ax.set_ylabel("Sample quantiles")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        n = len(resid)
        if n <= 5000:
            sw_stat, sw_p = shapiro(resid)
            sw_s = f"{sw_stat:.4f}"
            sw_ps = f"{sw_p:.4f}"
        else:
            sw_s, sw_ps, sw_p = "n/a", "n/a", 1.0

        ks_stat, ks_p = kstest(resid, "norm", args=(resid.mean(), resid.std()))
        verdict = "normal" if (sw_p > 0.05 if n <= 5000 else ks_p > 0.05) else "non-normal"
        print(f"  {mat:<12}  {n:>6}  {sw_s:>10}  {sw_ps:>10}  "
              f"{ks_stat:>8.4f}  {ks_p:>8.4f}  {verdict:>10}")

    plt.tight_layout()
    plt.savefig("appendix_qq_plots.pdf", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: appendix_qq_plots.pdf")

    # ── Paper validation — Task 4: Stage 1 reliability diagrams ──────────────
    proba_test = model.stage1.predict_proba(data["X_test_proc"])
    y_test_pres_bin = (data["y_test_mask"] & (data["y_test_raw"] > 0)).astype(int)

    N_BINS = 10
    fig, axes = plt.subplots(1, len(y_cols), figsize=(3.5 * len(y_cols), 3.5))
    fig.suptitle(
        "Stage 1 Reliability Diagrams (Calibration Curves) — Test Set\n"
        "CalibratedClassifierCV(method='sigmoid'); closer to diagonal = better calibrated",
        fontsize=10,
    )

    print(f"{'Material':<12}  {'ECE':>7}  {'MCE':>7}  {'Brier':>8}  {'Verdict':>12}")
    print("-" * 54)

    for ax, (m, mat) in zip(axes, enumerate(y_cols)):
        p_pred = proba_test[:, m]
        y_true = y_test_pres_bin[:, m].astype(float)

        prob_true, prob_pred = calibration_curve(
            y_true, p_pred, n_bins=N_BINS, strategy="uniform"
        )

        bins  = np.linspace(0, 1, N_BINS + 1)
        ece_sum = 0.0; mce = 0.0
        n_tot = len(y_true)
        for b in range(N_BINS):
            in_bin = (p_pred >= bins[b]) & (p_pred < bins[b + 1])
            if in_bin.sum() == 0:
                continue
            acc = float(y_true[in_bin].mean())
            conf = float(p_pred[in_bin].mean())
            err = abs(acc - conf)
            ece_sum += in_bin.sum() * err
            mce = max(mce, err)
        ece = ece_sum / n_tot
        brier = float(np.mean((p_pred - y_true) ** 2))

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.plot(prob_pred, prob_true, "o-", ms=5, lw=1.5, color="#e36c09", label="Model")
        ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.15, color="#e36c09")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{mat}\nECE={ece:.3f}  Brier={brier:.3f}", fontsize=9)
        ax.set_xlabel("Mean predicted prob.")
        if m == 0:
            ax.set_ylabel("Fraction of positives")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        verdict = "well-calibrated" if ece < 0.05 else ("acceptable" if ece < 0.10 else "poor")
        print(f"  {mat:<12}  {ece:>7.4f}  {mce:>7.4f}  {brier:>8.4f}  {verdict:>12}")

    plt.tight_layout()
    plt.savefig("paper_reliability_diagrams.pdf", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: paper_reliability_diagrams.pdf")
    print("\nECE (Expected Calibration Error) < 0.05 indicates good calibration.")

    # ── Paper validation — Task 5: 5-fold cross-validation ───────────────────
    X_cv_raw  = pd.concat([data["X_train_raw"], data["X_val_raw"]], ignore_index=True)
    y_cv_raw  = np.vstack([data["y_train_raw"], data["y_val_raw"]])
    y_cv_mask = np.vstack([data["y_train_mask"], data["y_val_mask"]])

    N_FOLDS = 5
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_maes  = {mat: [] for mat in y_cols}
    fold_crps  = {mat: [] for mat in y_cols}
    fold_prese = {mat: [] for mat in y_cols}

    print(f"Running {N_FOLDS}-fold cross-validation on train+val data "
          f"(n={len(X_cv_raw)} rows)...\n")

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_cv_raw)):
        print(f"  Fold {fold_idx + 1}/{N_FOLDS}  (train={len(tr_idx)}, val={len(va_idx)}) ... ",
              end="", flush=True)

        X_tr_raw = X_cv_raw.iloc[tr_idx].reset_index(drop=True)
        X_va_raw = X_cv_raw.iloc[va_idx].reset_index(drop=True)
        y_tr_raw = y_cv_raw[tr_idx]
        y_va_raw = y_cv_raw[va_idx]
        y_tr_msk = y_cv_mask[tr_idx]
        y_va_msk = y_cv_mask[va_idx]

        fold_pre = ColumnTransformer(transformers=[
            ("num", StandardScaler(), ["Construction period"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["Typology", "Primary Code", "Hybrid Structure", "Country"]),
        ])
        X_tr_proc = fold_pre.fit_transform(X_tr_raw)
        X_va_proc = fold_pre.transform(X_va_raw)

        fold_model = TwoStageConditionalModel(random_state=SEED)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_model.fit(X_tr_proc, X_tr_raw, y_tr_raw, y_tr_msk)

        for m, mat in enumerate(y_cols):
            va_pres = y_va_msk[:, m] & (y_va_raw[:, m] > 0)
            if va_pres.sum() < 3:
                continue

            qxgb = fold_model.stage2.models_.get(mat)
            if qxgb is None:
                continue

            y_true = y_va_raw[va_pres, m]
            y_log  = np.log(y_true + LOG_EPS)
            pq     = qxgb.predict_q_log(X_va_proc[va_pres])
            p50    = pq[:, 1]

            mae  = float(np.mean(np.abs(y_true - (np.exp(p50) - LOG_EPS))))
            crps = qxgb.crps_gaussian(X_va_proc[va_pres], y_log)
            fold_maes[mat].append(mae)
            fold_crps[mat].append(crps)

        pres_samp = fold_model.stage1.sample_presence(
            X_va_proc, n_samples=200, temperature=2.5, random_state=SEED,
        )
        samp_freq = pres_samp.mean(axis=1)
        true_freq_fold = (y_va_msk & (y_va_raw > 0)).astype(float)
        for m, mat in enumerate(y_cols):
            err = float(np.abs(samp_freq[:, m].mean() - true_freq_fold[:, m].mean()))
            fold_prese[mat].append(err)

        print("done")

    print(f"\n{'Material':<12}  {'MAE mean':>10}  {'MAE std':>9}  "
          f"{'CRPS mean':>10}  {'CRPS std':>9}  {'Pres |err|':>10}")
    print("-" * 68)

    cv_summary = []
    for mat in y_cols:
        maes = fold_maes[mat]
        crps = fold_crps[mat]
        pres = fold_prese[mat]
        if not maes:
            print(f"  {mat:<12}  {'—':>10}")
            continue
        mae_m  = float(np.mean(maes));  mae_s  = float(np.std(maes))
        crps_m = float(np.mean(crps));  crps_s = float(np.std(crps))
        pres_m = float(np.mean(pres))
        cv_summary.append(dict(
            Material=mat,
            MAE_mean=round(mae_m, 2), MAE_std=round(mae_s, 2),
            CRPS_mean=round(crps_m, 4), CRPS_std=round(crps_s, 4),
            Pres_err=round(pres_m, 3),
        ))
        print(f"  {mat:<12}  {mae_m:>10.2f}  {mae_s:>9.2f}  "
              f"{crps_m:>10.4f}  {crps_s:>9.4f}  {pres_m:>10.3f}")

    print()
    cv_df = pd.DataFrame(cv_summary)
    print("All figures in kg/m² (MAE) or log-kg/m² (CRPS).")
    print("Pres |err| = |sampled presence freq − true freq|.")

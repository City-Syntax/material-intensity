import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier, XGBRegressor

SEED = 42
LOG_EPS = 1e-6
Y_COLS = ["Concrete", "Glass", "Steel", "Wood", "Brick"]
GROUP_COLS = ["Typology"]


def build_group_keys(df, group_cols=GROUP_COLS):
    return (
        df.loc[:, group_cols]
        .fillna("Missing")
        .astype(str)
        .agg(" | ".join, axis=1)
        .to_numpy()
    )


class MaterialOccurrenceModel:
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

    def fit(self, X, y_presence):
        self.models_ = {}
        self.trivial_proba_ = {}
        for m, material in enumerate(Y_COLS):
            obs = y_presence[:, m]
            p = float(obs.mean())
            if p == 0.0 or p == 1.0:
                self.trivial_proba_[material] = p
                self.models_[material] = None
                continue
            clf = XGBClassifier(**self.xgb_params)
            clf.fit(X, obs.astype(int))
            self.models_[material] = clf
        return self

    def predict_proba(self, X):
        n_rows = X.shape[0]
        proba = np.zeros((n_rows, len(Y_COLS)), dtype=np.float64)
        for m, material in enumerate(Y_COLS):
            model = self.models_.get(material)
            if model is None:
                proba[:, m] = self.trivial_proba_.get(material, 1.0)
            else:
                proba[:, m] = model.predict_proba(X)[:, 1]
        return proba


class _PerMaterialMoE:
    """Mixture-of-Experts intensity regressor for a single material (log-space).

    Fits K latent regimes on log-transformed targets via GaussianMixture, trains
    one XGBoost expert per regime plus a gating XGBClassifier.  Quantile
    inference uses the law of total variance over the Gaussian mixture, which
    produces wider intervals when the gating is uncertain (the key benefit for
    multimodal materials like Concrete and Brick).
    """

    def __init__(self, n_components, xgb_params, random_state):
        self.n_components = n_components
        self.xgb_params = xgb_params
        self.random_state = random_state

    def fit(self, X, y_log):
        n = len(y_log)
        # Require at least 5 samples per component to avoid degenerate splits.
        K = max(1, min(self.n_components, n // 5))

        gm = GaussianMixture(
            n_components=K, random_state=self.random_state, n_init=3, max_iter=300
        )
        gm.fit(y_log.reshape(-1, 1))
        raw_labels = gm.predict(y_log.reshape(-1, 1))

        # Compact labels to 0..K'-1 in case some GM components are empty.
        unique, raw_labels = np.unique(raw_labels, return_inverse=True)
        K = len(unique)
        self.K_ = K

        # Gating classifier: P(regime | X).
        if K == 1:
            self.gate_ = None
        else:
            gate_kw = {k: v for k, v in self.xgb_params.items() if k != "objective"}
            self.gate_ = XGBClassifier(**gate_kw)
            self.gate_.fit(X, raw_labels)

        # One expert per regime + per-regime residual std.
        self.experts_ = {}
        self.expert_sigmas_ = np.zeros(K)
        for k in range(K):
            mask = raw_labels == k
            if mask.sum() < 2:
                mask = np.ones(n, dtype=bool)
            reg = XGBRegressor(**self.xgb_params)
            reg.fit(X[mask], y_log[mask])
            self.experts_[k] = reg
            resids = y_log[mask] - reg.predict(X[mask])
            self.expert_sigmas_[k] = max(float(np.std(resids)), 1e-6)

        return self

    def _weights(self, X):
        """Returns gate probabilities as (n_samples, K) array."""
        n, K = X.shape[0], self.K_
        if self.gate_ is None:
            return np.ones((n, 1))
        w = np.zeros((n, K))
        proba = self.gate_.predict_proba(X)
        for col, cls in enumerate(self.gate_.classes_):
            w[:, int(cls)] = proba[:, col]
        return w

    def predict_log_mean(self, X):
        """Mixture mean in log-space (used by JointDistributionModel)."""
        w = self._weights(X)                                               # (n, K)
        mu = np.stack([self.experts_[k].predict(X) for k in range(self.K_)], axis=1)
        return (w * mu).sum(axis=1)                                        # (n,)

    def predict_quantiles(self, X, q_lo=0.05, q_hi=0.95):
        """Mixture quantiles via law of total variance + Gaussian approx.

        Returns (p_lo, p50, p_hi) as numpy arrays in original (non-log) space.
        Mixture variance = E[Var[Y|Z]] (within-regime) + Var[E[Y|Z]] (between-
        regime), so uncertain gating automatically inflates the interval width.
        """
        K = self.K_
        w = self._weights(X)                                               # (n, K)
        mu = np.stack([self.experts_[k].predict(X) for k in range(K)], axis=1)
        sigma_k = self.expert_sigmas_                                      # (K,)

        mu_mix = (w * mu).sum(axis=1)                                      # (n,)
        within_var = (w * sigma_k[np.newaxis, :] ** 2).sum(axis=1)
        between_var = (w * (mu - mu_mix[:, np.newaxis]) ** 2).sum(axis=1)
        sigma_mix = np.sqrt(np.maximum(within_var + between_var, 1e-12))   # (n,)

        p_lo = np.maximum(np.exp(mu_mix + stats.norm.ppf(q_lo) * sigma_mix) - LOG_EPS, 0.0)
        p50  = np.maximum(np.exp(mu_mix) - LOG_EPS, 0.0)
        p_hi = np.maximum(np.exp(mu_mix + stats.norm.ppf(q_hi) * sigma_mix) - LOG_EPS, 0.0)
        return p_lo, p50, p_hi


class MaterialIntensityModel:
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
        n_components=3,
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
            objective="reg:squarederror",
            verbosity=0,
        )
        self.n_components = n_components
        self.random_state = random_state
        self.models_ = {}

    def fit(self, X, y_raw, y_presence):
        self.models_ = {}
        for m, material in enumerate(Y_COLS):
            obs = y_presence[:, m]
            if obs.sum() < 2:
                self.models_[material] = None
                continue
            y_log = np.log(y_raw[obs, m] + LOG_EPS)
            moe = _PerMaterialMoE(
                n_components=self.n_components,
                xgb_params=self.xgb_params,
                random_state=self.random_state,
            )
            moe.fit(X[obs], y_log)
            self.models_[material] = moe
        return self

    def predict_log(self, X):
        """Mixture mean in log-space — backward-compatible with JointDistributionModel."""
        n_rows = X.shape[0]
        mu_log = np.zeros((n_rows, len(Y_COLS)), dtype=np.float64)
        for m, material in enumerate(Y_COLS):
            moe = self.models_.get(material)
            if moe is not None:
                mu_log[:, m] = moe.predict_log_mean(X)
        return mu_log

    def predict(self, X):
        return np.maximum(np.exp(self.predict_log(X)) - LOG_EPS, 0.0)

    def predict_intervals(self, X, alpha=0.10):
        """MoE quantile intervals.  Output: {material: {'p5', 'p50', 'p95'}}."""
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
        result = {}
        n = X.shape[0]
        for material in Y_COLS:
            moe = self.models_.get(material)
            if moe is None:
                result[material] = {
                    "p5": np.zeros(n),
                    "p50": np.zeros(n),
                    "p95": np.zeros(n),
                }
            else:
                p_lo, p50, p_hi = moe.predict_quantiles(X, q_lo=q_lo, q_hi=q_hi)
                result[material] = {"p5": p_lo, "p50": p50, "p95": p_hi}
        return result


class JointDistributionModel:
    def __init__(self, group_cols=GROUP_COLS, min_group_size=20, reg_eps=1e-4, cov_shrink=0.0):
        self.group_cols = tuple(group_cols)
        self.min_group_size = min_group_size
        self.reg_eps = reg_eps
        self.cov_shrink = float(np.clip(cov_shrink, 0.0, 1.0))
        self.global_cov_ = None
        self.group_covs_ = {}

    def _regularise_cov(self, cov):
        diag_cov = np.diag(np.diag(cov))
        shrunk = (1.0 - self.cov_shrink) * cov + self.cov_shrink * diag_cov
        eye_m = np.eye(len(Y_COLS))
        return shrunk + eye_m * self.reg_eps

    def fit(self, X_proc, X_raw, y_raw, y_presence, intensity_model):
        # Complete cases: all five materials are truly present (observed AND > 0).
        # Using y_mask.all() would include observed-as-zero rows, whose log(0+eps)
        # values contaminate the residual covariance with near-−∞ entries.
        complete = y_presence.all(axis=1)
        if complete.sum() < len(Y_COLS):
            raise ValueError(f"Need >= {len(Y_COLS)} presence-complete rows; got {complete.sum()}.")

        mu_log = intensity_model.predict_log(X_proc)
        y_log = np.log(y_raw[complete] + LOG_EPS)
        residuals = y_log - mu_log[complete]
        groups = build_group_keys(X_raw.loc[complete].reset_index(drop=True), self.group_cols)

        eye_m = np.eye(len(Y_COLS))
        self.global_cov_ = (
            self._regularise_cov(np.cov(residuals, rowvar=False))
            if residuals.shape[0] >= len(Y_COLS)
            else eye_m * self.reg_eps
        )
        self.group_covs_ = {}
        for g in np.unique(groups):
            g_res = residuals[groups == g]
            if g_res.shape[0] >= self.min_group_size:
                self.group_covs_[g] = self._regularise_cov(np.cov(g_res, rowvar=False))
        return self

    def get_cov(self, group):
        return self.group_covs_.get(group, self.global_cov_)

    def predict_intervals(self, X_proc, groups, intensity_model, alpha=0.10):
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        mu_log = intensity_model.predict_log(X_proc)
        unique_groups = np.unique(groups)
        sigma_cache = {g: np.sqrt(np.diag(self.get_cov(g))) for g in unique_groups}
        sigma = np.stack([sigma_cache[g] for g in groups])

        lo_log = mu_log - z * sigma
        hi_log = mu_log + z * sigma
        p_lo = np.maximum(np.exp(lo_log) - LOG_EPS, 0.0)
        p50 = np.maximum(np.exp(mu_log) - LOG_EPS, 0.0)
        p_hi = np.maximum(np.exp(hi_log) - LOG_EPS, 0.0)

        return {
            mat: {"p5": p_lo[:, m], "p50": p50[:, m], "p95": p_hi[:, m]}
            for m, mat in enumerate(Y_COLS)
        }


class TwoStageConditionalModel:
    def __init__(
        self,
        group_cols=GROUP_COLS,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_group_size=20,
        reg_eps=1e-4,
        cov_shrink=0.0,
        random_state=SEED,
        n_components=3,
    ):
        xgb_kw = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
        )
        self.stage1 = MaterialOccurrenceModel(**xgb_kw)
        self.stage2 = MaterialIntensityModel(**xgb_kw, n_components=n_components)
        self.joint = JointDistributionModel(
            group_cols=group_cols,
            min_group_size=min_group_size,
            reg_eps=reg_eps,
            cov_shrink=cov_shrink,
        )

    def fit(self, X_proc, X_raw, y_raw, y_mask):
        # y_presence: reported AND positive — the true "material exists" signal
        y_presence = y_mask & (y_raw > 0)
        self.stage1.fit(X_proc, y_presence)
        self.stage2.fit(X_proc, y_raw, y_presence)
        self.joint.fit(X_proc, X_raw, y_raw, y_presence, self.stage2)
        return self

    def predict(self, X_proc, groups, alpha=0.10):
        proba = self.stage1.predict_proba(X_proc)
        intervals = self.stage2.predict_intervals(X_proc, alpha=alpha)
        for m, mat in enumerate(Y_COLS):
            intervals[mat]["p_presence"] = proba[:, m]
        return intervals

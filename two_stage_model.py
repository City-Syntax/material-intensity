import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, RidgeCV

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
    def __init__(self, C=1.0, max_iter=1000, random_state=SEED):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.models_ = {}
        self.trivial_proba_ = {}

    def fit(self, X, y_mask):
        self.models_ = {}
        self.trivial_proba_ = {}
        for m, material in enumerate(Y_COLS):
            obs = y_mask[:, m]
            p = float(obs.mean())
            if p == 0.0 or p == 1.0:
                self.trivial_proba_[material] = p
                self.models_[material] = None
                continue
            lr = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver="lbfgs",
            )
            lr.fit(X, obs.astype(int))
            self.models_[material] = lr
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


class MaterialIntensityModel:
    def __init__(self, alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)):
        self.alphas = alphas
        self.models_ = {}

    def fit(self, X, y_raw, y_mask):
        self.models_ = {}
        for m, material in enumerate(Y_COLS):
            obs = y_mask[:, m]
            if obs.sum() < 2:
                self.models_[material] = None
                continue
            X_obs = X[obs]
            y_log = np.log(y_raw[obs, m] + LOG_EPS)
            ridge = RidgeCV(alphas=list(self.alphas), cv=5)
            ridge.fit(X_obs, y_log)
            self.models_[material] = ridge
        return self

    def predict_log(self, X):
        n_rows = X.shape[0]
        mu_log = np.zeros((n_rows, len(Y_COLS)), dtype=np.float64)
        for m, material in enumerate(Y_COLS):
            model = self.models_.get(material)
            if model is not None:
                mu_log[:, m] = model.predict(X)
        return mu_log

    def predict(self, X):
        return np.maximum(np.exp(self.predict_log(X)) - LOG_EPS, 0.0)


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

    def fit(self, X_proc, X_raw, y_raw, y_mask, intensity_model):
        complete = y_mask.all(axis=1)
        if complete.sum() < len(Y_COLS):
            raise ValueError(f"Need >= {len(Y_COLS)} complete rows; got {complete.sum()}.")

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
        logistic_C=1.0,
        ridge_alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
        min_group_size=20,
        reg_eps=1e-4,
        cov_shrink=0.0,
        random_state=SEED,
    ):
        self.stage1 = MaterialOccurrenceModel(C=logistic_C, random_state=random_state)
        self.stage2 = MaterialIntensityModel(alphas=ridge_alphas)
        self.joint = JointDistributionModel(
            group_cols=group_cols,
            min_group_size=min_group_size,
            reg_eps=reg_eps,
            cov_shrink=cov_shrink,
        )

    def fit(self, X_proc, X_raw, y_raw, y_mask):
        self.stage1.fit(X_proc, y_mask)
        self.stage2.fit(X_proc, y_raw, y_mask)
        self.joint.fit(X_proc, X_raw, y_raw, y_mask, self.stage2)
        return self

    def predict(self, X_proc, groups, alpha=0.10):
        proba = self.stage1.predict_proba(X_proc)
        intervals = self.joint.predict_intervals(X_proc, groups, self.stage2, alpha=alpha)
        for m, mat in enumerate(Y_COLS):
            intervals[mat]["p_presence"] = proba[:, m]
        return intervals

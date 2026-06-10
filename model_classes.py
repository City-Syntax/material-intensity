import random
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

SEED = 42
LOW_OBS_THRESHOLD = 30
IPS_MIN_PROBA = 0.05
IPS_MAX_WEIGHT = 20.0
ARCHETYPE_HIGH_SUPPORT = 20
ARCHETYPE_MEDIUM_SUPPORT = 8
ARCHETYPE_LOW_SUPPORT = 3

archetype_cols = [
    "Construction period bucket",
    "Typology",
    "Primary Code",
    "Country_imputed",
]
y_cols = ["Concrete", "Glass", "Steel", "Wood", "Brick"]


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


class ObservationModel:
    """Per-material XGBoost classifier predicting P(recorded | x)."""

    def __init__(self, n_estimators=300, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.0, reg_lambda=1.0, random_state=SEED):
        self.xgb_params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
            reg_lambda=reg_lambda, random_state=random_state,
            objective="binary:logistic", verbosity=0,
        )
        self.models_ = {}
        self.trivial_proba_ = {}

    def fit(self, X, y_observed):
        """Fit one CalibratedClassifierCV per material.

        Parameters
        ----------
        X          : (n, d) preprocessed feature matrix
        y_observed : (n, M) bool  --  True where intensity was recorded
        """
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
        """Return (n, M) array of P(recorded | x) for each material."""
        n = X.shape[0]
        proba = np.zeros((n, len(y_cols)), dtype=np.float64)
        for m, material in enumerate(y_cols):
            clf = self.models_.get(material)
            if clf is None:
                proba[:, m] = self.trivial_proba_.get(material, 0.0)
            else:
                proba[:, m] = clf.predict_proba(X)[:, 1]
        return proba



class IntensityModel:
    """Per-material dual-head XGBoost estimating conditional intensity (kg/m^2)."""

    ALPHAS = [0.05, 0.50, 0.95]

    def __init__(self, n_estimators=300, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.0, reg_lambda=1.0, random_state=SEED,
                 per_material_params=None):
        self.xgb_params = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
            reg_lambda=reg_lambda, random_state=random_state,
            verbosity=0,
        )
        self.per_material_params = per_material_params or {}
        self.models_ = {}
        self.mean_models_ = {}

    @staticmethod
    def _inverse_propensity_weights(p_recorded_obs):
        p = np.clip(np.asarray(p_recorded_obs, dtype=np.float64), IPS_MIN_PROBA, 1.0)
        w = 1.0 / p
        w = np.minimum(w, IPS_MAX_WEIGHT)
        # Keep mean weight near 1.0 for stable optimization across materials.
        return w / np.mean(w)

    def fit(self, X, y_raw, y_observed, p_recorded=None):
        """Train quantile and mean XGBoost per material on observed rows.

        Parameters
        ----------
        X          : (n, d) preprocessed feature matrix
        y_raw      : (n, M) float  --  intensity values (NaN where unobserved)
        y_observed : (n, M) bool   --  True where intensity was recorded
        p_recorded : (n, M) float  --  Stage 1 P(recorded|x), used only for IPS weighting
        """
        self.models_ = {}
        self.mean_models_ = {}
        for m, material in enumerate(y_cols):
            obs = y_observed[:, m]
            if obs.sum() < 2:
                self.models_[material] = None
                self.mean_models_[material] = None
                continue
            y_log = np.log1p(y_raw[obs, m])
            if p_recorded is None:
                sw = np.ones(obs.sum(), dtype=np.float64)
            else:
                sw = self._inverse_propensity_weights(p_recorded[obs, m])

            # Merge global params with any per-material overrides
            mat_params = {**self.xgb_params, **self.per_material_params.get(material, {})}

            # Head 1: Quantile regression
            mdl = XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=self.ALPHAS,
                **mat_params,
            )
            mdl.fit(X[obs], y_log, sample_weight=sw)
            self.models_[material] = mdl

            # Head 2: Mean prediction (L2)
            mean_mdl = XGBRegressor(
                objective="reg:squarederror",
                **mat_params,
            )
            mean_mdl.fit(X[obs], y_log, sample_weight=sw)
            self.mean_models_[material] = mean_mdl
        return self

    def predict_quantiles(self, X):
        """Return p05/p50/p95 in original (kg/m^2) space per material.

        Intervals are the direct expm1-transformed outputs of the quantile
        regression model (quantile_alpha = [0.05, 0.50, 0.95]).
        Inverse of log1p is expm1; no clipping or Gaussian approximation is used.

        Returns dict: material -> {"p05": array, "p50": array, "p95": array}
        """
        n = X.shape[0]
        result = {}
        for material in y_cols:
            mdl = self.models_.get(material)
            if mdl is None:
                result[material] = {"p05": np.zeros(n), "p50": np.zeros(n), "p95": np.zeros(n)}
                continue
            pq   = mdl.predict(X)      # (n, 3): [log1p_p05, log1p_p50, log1p_p95]
            p_lo = np.maximum(np.expm1(pq[:, 0]), 0.0)
            p50  = np.maximum(np.expm1(pq[:, 1]), 0.0)
            p_hi = np.maximum(np.expm1(pq[:, 2]), 0.0)
            result[material] = {"p05": p_lo, "p50": p50, "p95": p_hi}
        return result

    def predict_means(self, X):
        """Return mean prediction in original (kg/m^2) space per material.

        Returns dict: material -> array of mean predictions
        """
        n = X.shape[0]
        result = {}
        for material in y_cols:
            mdl = self.mean_models_.get(material)
            if mdl is None:
                result[material] = np.zeros(n)
                continue
            y_mean_log = mdl.predict(X)
            y_mean = np.maximum(np.expm1(y_mean_log), 0.0)
            result[material] = y_mean
        return result



class FinalQueryModel:
    """Two-stage database-informed query model.

    Stage 1 (tuned_observation_model): P(recorded | x) per material.
    Stage 2 (tuned_intensity_model):   conditional intensity quantiles + mean (kg/m^2).
      Quantiles at [0.05, 0.50, 0.95] via quantile regression.
      Mean via standard L2 regression.
      Per-material hyperparameters via per_material_intensity_params.

    Archetype-mean blending: for buildings whose archetype (archetype_cols tuple)
    appears in training data, p50 and mean predictions are blended toward the
    empirical archetype mean.  Blend weight grows linearly from 0 (no support)
    to BLEND_MAX_ALPHA at ARCHETYPE_HIGH_SUPPORT rows, then stays capped.
    p05/p95 interval bounds are not shifted.

    query() output per material
    ---------------------------
    database_reporting_probability   float   P(recorded | x)
    p_recorded                       float   alias of database_reporting_probability
    p05, p50, p95                    float   conditional intensity quantiles (kg/m^2)
    mean                             float   conditional intensity mean (kg/m^2)
    expected_reported                float   p_recorded * p50
    n_observed_train                 int     training rows with this material recorded
    coverage_warning                 bool    True if n_observed_train < LOW_OBS_THRESHOLD
    archetype_n_train                int     training rows matching the full input archetype
    archetype_support_level          str     qualitative support label from archetype_n_train
    """

    BLEND_MAX_ALPHA = 0.0  # blending disabled; set > 0 to re-enable archetype-mean blending

    def __init__(self, observation_params=None, intensity_params=None,
                 per_material_intensity_params=None):
        _default = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.0, reg_lambda=1.0, random_state=SEED)
        self.tuned_observation_model = ObservationModel(**(observation_params or _default))
        self.tuned_intensity_model   = IntensityModel(
            **(intensity_params or _default),
            per_material_params=per_material_intensity_params,
        )

    @staticmethod
    def _archetype_key_row(row):
        return tuple("<NA>" if pd.isna(v) else str(v) for v in row)

    def _build_archetype_count_map(self, X_raw):
        if X_raw is None:
            return {}
        xdf = pd.DataFrame(X_raw).copy()
        xdf = xdf[archetype_cols].copy()
        keys = xdf.apply(self._archetype_key_row, axis=1)
        counts = keys.value_counts()
        return {k: int(v) for k, v in counts.items()}

    def _build_archetype_means(self, X_raw, y_raw, y_mask):
        """Per-archetype, per-material mean intensity (original scale) from training."""
        xdf = X_raw[archetype_cols].copy().reset_index(drop=True)
        keys = xdf.apply(self._archetype_key_row, axis=1)
        # Group by archetype key with a plain dict to avoid numpy
        # broadcasting errors when comparing arrays of tuples.
        key_to_rows = {}
        for i, k in enumerate(keys):
            key_to_rows.setdefault(k, []).append(i)
        archetype_means = {}
        for key, row_indices in key_to_rows.items():
            row_idx = np.array(row_indices)
            archetype_means[key] = {}
            for m in range(len(y_cols)):
                obs = y_mask[row_idx, m]
                if obs.sum() >= 1:
                    archetype_means[key][m] = float(np.mean(y_raw[row_idx[obs], m]))
                else:
                    archetype_means[key][m] = np.nan
        return archetype_means

    def fit(self, X_proc, y_raw, y_mask, X_raw=None):
        y_observed = y_mask
        self.n_observed_train_ = {
            mat: int(y_observed[:, m].sum()) for m, mat in enumerate(y_cols)
        }
        self.archetype_count_map_ = self._build_archetype_count_map(X_raw)
        self.archetype_means_ = (
            self._build_archetype_means(X_raw, y_raw, y_mask)
            if X_raw is not None else {}
        )
        self.tuned_observation_model.fit(X_proc, y_observed)
        p_recorded = self.tuned_observation_model.predict_proba(X_proc)
        self.tuned_intensity_model.fit(X_proc, y_raw, y_observed, p_recorded=p_recorded)
        return self

    def query(self, X_proc, X_raw=None):
        """Query the model for one or more buildings.

        Parameters
        ----------
        X_proc : (n, d) preprocessed feature matrix
        X_raw  : (n, len(X_cols)) raw feature rows for archetype support matching

        Returns
        -------
        dict: material -> {database_reporting_probability, p_recorded, p05, p50, p95, mean,
                           expected_reported, n_observed_train, coverage_warning,
                           archetype_n_train, archetype_support_level}
        """
        p_recorded = self.tuned_observation_model.predict_proba(X_proc)
        intervals  = self.tuned_intensity_model.predict_quantiles(X_proc)
        means_dict = self.tuned_intensity_model.predict_means(X_proc)

        n = X_proc.shape[0]
        if X_raw is None or not getattr(self, "archetype_count_map_", None):
            archetype_n   = np.full(n, np.nan)
            archetype_lvl = np.array(["unknown"] * n, dtype=object)
        else:
            xdf = pd.DataFrame(X_raw).copy()
            xdf = xdf[archetype_cols].copy()
            keys = xdf.apply(self._archetype_key_row, axis=1)
            archetype_n = np.array(
                [self.archetype_count_map_.get(k, 0) for k in keys],
                dtype=np.int64,
            )
            archetype_lvl = np.array(
                [archetype_support_level(int(v)) for v in archetype_n], dtype=object
            )

            # Blend p50 and mean toward archetype training mean, weighted by support.
            # Blending starts only at ARCHETYPE_MEDIUM_SUPPORT (8 rows) and grows
            # linearly to BLEND_MAX_ALPHA at ARCHETYPE_HIGH_SUPPORT (20 rows).
            # p05/p95 interval bounds are not shifted.
            _eff_n  = np.maximum(archetype_n - ARCHETYPE_MEDIUM_SUPPORT, 0)
            _scale  = ARCHETYPE_HIGH_SUPPORT - ARCHETYPE_MEDIUM_SUPPORT  # 20 - 8 = 12
            blend_w = np.clip(_eff_n / _scale, 0.0, 1.0) * self.BLEND_MAX_ALPHA
            arch_keys = keys.values

            for m, mat in enumerate(y_cols):
                arch_means_arr = np.array(
                    [self.archetype_means_.get(k, {}).get(m, np.nan) for k in arch_keys],
                    dtype=np.float64,
                )
                valid = np.isfinite(arch_means_arr) & (blend_w > 0)
                if valid.any():
                    w = blend_w[valid]
                    intervals[mat]["p50"][valid] = (
                        (1.0 - w) * intervals[mat]["p50"][valid]
                        + w * arch_means_arr[valid]
                    )
                    means_dict[mat][valid] = (
                        (1.0 - w) * means_dict[mat][valid]
                        + w * arch_means_arr[valid]
                    )

        result = {}
        for m, mat in enumerate(y_cols):
            n_obs = self.n_observed_train_.get(mat, 0)
            result[mat] = {
                "database_reporting_probability": p_recorded[:, m],
                "p_recorded":                      p_recorded[:, m],
                "p05":                             intervals[mat]["p05"],
                "p50":                             intervals[mat]["p50"],
                "p95":                             intervals[mat]["p95"],
                "mean":                            means_dict[mat],
                "expected_reported":               p_recorded[:, m] * intervals[mat]["p50"],
                "n_observed_train":                n_obs,
                "coverage_warning":                n_obs < LOW_OBS_THRESHOLD,
                "archetype_n_train":               archetype_n,
                "archetype_support_level":         archetype_lvl,
            }
        return result



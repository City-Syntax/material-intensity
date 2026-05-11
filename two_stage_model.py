import numpy as np
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier, XGBRegressor

SEED = 42
LOG_EPS = 1e-6
Y_COLS = ["Concrete", "Glass", "Steel", "Wood", "Brick"]
GROUP_COLS = ["Primary Code"]

# Dominant structural material per Primary Code → minimum presence probability during sampling.
# Applied after isotonic calibration, before the final probability clip.
STRUCTURAL_PRIOR = {
    "B": {"Brick":    0.7},
    "C": {"Concrete": 0.7},
    "S": {"Steel":    0.7},
    "W": {"Wood":     0.7},
    # "T" (Traditional): no single dominant material — omitted
}

# ── Sampling hyperparameters ────────────────────────────────────────────────
BLEND_LAMBDA   = 0.35   # chain weight in blend:  p = λ·p_chain + (1-λ)·p_marginal
DROPOUT_RATE   = 0.08   # diversity dropout: flip "present" → "absent" with this prob
P_CLIP_LO      = 0.05   # presence probability hard lower clip (prevents probability 0)
P_CLIP_HI      = 0.95   # presence probability hard upper clip (prevents probability 1)
RESIDUAL_SCALE = 0.35   # Stage 3 log-space residual amplitude
CROSS_DAMP     = 0.25   # Stage 3 off-diagonal covariance damping factor
Z_QUANT_95     = 1.6449 # norm.ppf(0.95); recovers sigma from [p05, p95] spread


def build_group_keys(df, group_cols=GROUP_COLS):
    return (
        df.loc[:, group_cols]
        .fillna("Missing")
        .astype(str)
        .agg(" | ".join, axis=1)
        .to_numpy()
    )


class MaterialOccurrenceModel:
    """Classifier chain for joint material presence modeling.

    Trains one CalibratedClassifierCV(XGBClassifier, method='sigmoid') per
    material in descending binary-entropy order.  Each classifier conditions on
    all previously fitted materials, capturing co-occurrence dependencies that
    independent Bernoulli sampling misses.
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
        self.chain_order_ = list(range(len(Y_COLS)))
        self.presence_calibrators_ = {}   # set by TwoStageConditionalModel.calibrate_sampling()

    @staticmethod
    def _chain_X(X, ctx_cols):
        """Append accumulated chain columns to base features."""
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
        ctx = []   # list of (N,) float arrays — observed labels so far
        for m in self.chain_order_:
            material = Y_COLS[m]
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
        """Marginal probabilities via greedy chain (Platt-calibrated)."""
        n = X.shape[0]
        proba = np.zeros((n, len(Y_COLS)), dtype=np.float64)
        ctx = []   # list of (n,) float arrays — hard predictions so far
        for m in self.chain_order_:
            material = Y_COLS[m]
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
        """Greedy chain predict_proba with a given material sampling order.

        Uses "available context": for each material at position k in perm, builds
        the training-order context columns, substituting zeros for any prior that
        has not yet been visited in this permutation.  This preserves the expected
        feature dimension of each chain model while allowing arbitrary ordering.
        """
        n         = X.shape[0]
        proba     = np.zeros((n, len(Y_COLS)))
        ctx_built = {}   # m (material index) -> (n,) hard-predicted float labels

        for k in range(len(Y_COLS)):
            m        = perm[k]
            material = Y_COLS[m]
            t        = train_pos[m]   # training chain position → expected ctx columns

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
        """Draw coherent material combinations via randomized ancestral chain sampling.

        Reduces fixed-order dependency artifacts by:
          • Averaging the blend-anchor marginal over n_chain_orders random orderings
          • Dividing n_samples into n_chain_orders groups each sampled with its own
            random chain order, using "available context" to preserve model dimensions
          • Residual-chain blend:  p = λ·p_chain + (1-λ)·p_marginal_avg  (λ=0.35)
          • Probability clip:      p = clip(p, 0.05, 0.95)
          • Diversity dropout:     flip "present" → "absent" with prob 0.08

        Parameters
        ----------
        X              : np.ndarray (n_rows, n_features) preprocessed features
        n_samples      : int  draws per query row
        temperature    : float > 0  logit temperature applied to p_chain before blend
        random_state   : int, Generator, or None
        n_chain_orders : int  random orderings to average for the marginal anchor
                         and to partition sample groups (default 4)

        Returns
        -------
        np.ndarray (n_rows, n_samples, n_materials) bool
        """
        _eps = 1e-9

        rng    = np.random.default_rng(random_state)
        n_rows = X.shape[0]
        M      = len(Y_COLS)
        out    = np.zeros((n_rows, n_samples, M), dtype=bool)

        # Precompute training position of each material index for O(1) lookup
        train_pos = {m: pos for pos, m in enumerate(self.chain_order_)}

        for i in range(n_rows):
            Xi = X[i:i+1]

            # ── Order-averaged marginal (blend anchor) ─────────────────────────
            # Average predict_proba over n_chain_orders orderings to remove the
            # fixed-order suppression/amplification that biases the blend anchor.
            p_avg = self.predict_proba(Xi)[0].copy()   # (M,) training order
            for _ in range(n_chain_orders - 1):
                perm  = rng.permutation(M)
                p_avg += self._predict_proba_order(Xi, perm, train_pos)[0]
            p_marginal = p_avg / n_chain_orders         # (M,) order-averaged

            # ── Stochastic draws split into n_chain_orders groups ──────────────
            # Each group uses its own random permutation so no single material is
            # always first (and thus always unconditioned).
            pres  = np.zeros((n_samples, M), dtype=bool)
            edges = np.linspace(0, n_samples, n_chain_orders + 1, dtype=int)

            for g in range(n_chain_orders):
                start, end = int(edges[g]), int(edges[g + 1])
                n_g   = end - start
                X_rep = np.tile(Xi, (n_g, 1))
                perm  = rng.permutation(M)        # random order for this group
                pres_g    = np.zeros((n_g, M), dtype=bool)
                ctx_built = {}                    # m -> (n_g,) sampled float labels

                for k in range(M):
                    m        = perm[k]
                    material = Y_COLS[m]
                    t        = train_pos[m]       # expected context columns

                    # "Available context": training-order priors, zeros if not yet
                    # sampled in this group's random order.
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

                    if self.presence_calibrators_.get(material) is not None:
                        p_m = self.presence_calibrators_[material].predict(p_m).clip(0.0, 1.0)

                    # Structural prior floor: primary structural material cannot collapse
                    if primary_codes is not None:
                        floor = STRUCTURAL_PRIOR.get(primary_codes[i], {}).get(material, 0.0)
                        if floor > 0.0:
                            p_m = np.maximum(p_m, floor)

                    p_m = np.clip(p_m, P_CLIP_LO, P_CLIP_HI)
                    z   = rng.random(n_g) < p_m
                    z   = z & (rng.random(n_g) >= DROPOUT_RATE)

                    pres_g[:, m] = z
                    ctx_built[m] = z.astype(np.float64)

                pres[start:end] = pres_g

            out[i] = pres

        return out

    def evaluate_calibration(self, X, y_presence, n_bins=10):
        """Reliability diagnostics for Stage 1 presence probabilities.

        Uses predict_proba output (Platt-calibrated via CalibratedClassifierCV)
        to compute per-material metrics and reliability curve data.

        Parameters
        ----------
        X          : np.ndarray (n_rows, n_features)
        y_presence : np.ndarray (n_rows, n_materials) bool  true presence
        n_bins     : int  probability bins for reliability curve (default 10)

        Returns
        -------
        dict[material -> dict] keys: mean_pred_bins, frac_pos_bins,
            brier, pred_mean, obs_freq
        """
        proba = self.predict_proba(X)
        print(f"  {'Material':<12}  {'Brier':>8}  {'Mean pred':>10}  "
              f"{'Obs freq':>10}  {'Bias':>8}")
        print("  " + "-" * 62)

        results = {}
        for m, material in enumerate(Y_COLS):
            y_true   = y_presence[:, m].astype(float)
            p_pred   = proba[:, m]

            brier    = float(np.mean((p_pred - y_true) ** 2))
            pred_mu  = float(p_pred.mean())
            obs_freq = float(y_true.mean())
            bias     = pred_mu - obs_freq

            bins = np.linspace(0.0, 1.0, n_bins + 1)
            mp_bins, fp_bins = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (p_pred >= lo) & (p_pred < hi)
                if mask.sum() > 0:
                    mp_bins.append(float(p_pred[mask].mean()))
                    fp_bins.append(float(y_true[mask].mean()))

            print(
                f"  {material:<12}  {brier:>8.4f}  {pred_mu:>9.3f}   "
                f"  {obs_freq:>9.3f}  {bias:>+8.3f}"
            )
            results[material] = dict(
                mean_pred_bins = np.array(mp_bins),
                frac_pos_bins  = np.array(fp_bins),
                brier          = brier,
                pred_mean      = pred_mu,
                obs_freq       = obs_freq,
            )
        return results


class _PerMaterialQuantileXGB:
    """Quantile XGBoost intensity regressor for a single material (log-space).

    Fits one XGBRegressor with objective='reg:quantileerror' at three levels
    [0.05, 0.50, 0.95].  Arbitrary quantiles use a Gaussian approximation
    whose sigma is recovered from the p05/p95 spread (z_{0.95} ≈ 1.6449).
    """

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
        """(n, 3) array of [p05, p50, p95] in log-space."""
        return self.model_.predict(X)

    def _sigma(self, pq):
        """Per-sample sigma from p05/p95 spread under Gaussian assumption."""
        return np.maximum((pq[:, 2] - pq[:, 0]) / (2 * Z_QUANT_95), 1e-6)

    def predict_log_mean(self, X):
        """Median in log-space (used by JointDistributionModel)."""
        return self.predict_q_log(X)[:, 1]

    def predict_quantiles(self, X, q_lo=0.05, q_hi=0.95):
        """Returns (p_lo, p50, p_hi) in original (non-log) space.

        Arbitrary q_lo / q_hi are computed via Gaussian approximation: sigma is
        recovered from the p05/p95 spread, then used with the normal PPF.
        """
        pq    = self.predict_q_log(X)                                      # (n, 3)
        mu    = pq[:, 1]
        sigma = self._sigma(pq)
        p_lo  = np.maximum(np.exp(mu + stats.norm.ppf(q_lo) * sigma) - LOG_EPS, 0.0)
        p50   = np.maximum(np.exp(mu) - LOG_EPS, 0.0)
        p_hi  = np.maximum(np.exp(mu + stats.norm.ppf(q_hi) * sigma) - LOG_EPS, 0.0)
        return p_lo, p50, p_hi

    def crps_gaussian(self, X, y_log):
        """Gaussian CRPS using quantile-inferred mu and sigma (log-space)."""
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
            verbosity=0,
        )
        self.models_ = {}

    def fit(self, X, y_raw, y_presence):
        self.models_ = {}
        for m, material in enumerate(Y_COLS):
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
        """Median in log-space — backward-compatible with JointDistributionModel."""
        n_rows = X.shape[0]
        mu_log = np.zeros((n_rows, len(Y_COLS)), dtype=np.float64)
        for m, material in enumerate(Y_COLS):
            qxgb = self.models_.get(material)
            if qxgb is not None:
                mu_log[:, m] = qxgb.predict_log_mean(X)
        return mu_log

    def predict(self, X):
        return np.maximum(np.exp(self.predict_log(X)) - LOG_EPS, 0.0)

    def predict_intervals(self, X, alpha=0.10):
        """Quantile intervals.  Output: {material: {'p5', 'p50', 'p95'}}."""
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
        result = {}
        n = X.shape[0]
        for material in Y_COLS:
            qxgb = self.models_.get(material)
            if qxgb is None:
                result[material] = {"p5": np.zeros(n), "p50": np.zeros(n), "p95": np.zeros(n)}
            else:
                p_lo, p50, p_hi = qxgb.predict_quantiles(X, q_lo=q_lo, q_hi=q_hi)
                result[material] = {"p5": p_lo, "p50": p50, "p95": p_hi}
        return result

    def evaluate_crps(self, X, y):
        """Print mean CRPS per material on held-out data (log-space).

        CRPS (Continuous Ranked Probability Score) evaluates the full predictive
        distribution, not just the point forecast.  Reported in log-space so values
        are scale-comparable across materials.
        """
        if hasattr(y, "to_numpy"):
            y = y.to_numpy(dtype=np.float64)

        print(f"  {'Material':<12}  {'n_obs':>6}  {'Mean CRPS (log-space)':>22}")
        print("  " + "-" * 44)
        for m, material in enumerate(Y_COLS):
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
        """Empirical coverage at each nominal level per material.

        For each nominal coverage level (default 10%–90% in steps of 10%),
        computes the fraction of held-out presence rows whose true value falls
        inside the symmetric predictive interval.  Returns a dict suitable for
        calibration plotting; also prints a table.
        """
        if hasattr(y, "to_numpy"):
            y = y.to_numpy(dtype=np.float64)
        if levels is None:
            levels = np.linspace(0.10, 0.90, 9)
        levels = np.asarray(levels)

        results = {}
        level_hdr = "  ".join(f"{int(round(lv * 100)):3d}%" for lv in levels)
        print(f"  {'Material':<12}  {'n_obs':>6}    {level_hdr}")
        print("  " + "-" * (22 + 7 * len(levels)))

        for m, material in enumerate(Y_COLS):
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


class JointDistributionModel:
    def __init__(self, group_cols=GROUP_COLS, min_group_size=20, reg_eps=1e-4, cov_shrink=0.0):
        self.group_cols = tuple(group_cols)
        self.min_group_size = min_group_size
        self.reg_eps = reg_eps
        self.cov_shrink = float(np.clip(cov_shrink, 0.0, 1.0))
        self.global_cov_ = None
        self.group_covs_ = {}

    def _pairwise_cov(self, residuals, presence):
        """Covariance matrix built pairwise: each (m1,m2) uses rows where both present."""
        M = len(Y_COLS)
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
        """Shrink toward ref (global cov for groups, diagonal for global), then add reg_eps * I."""
        if ref is None:
            ref = np.diag(np.diag(cov))
        shrunk = (1.0 - self.cov_shrink) * cov + self.cov_shrink * ref
        return shrunk + np.eye(len(Y_COLS)) * self.reg_eps

    def fit(self, X_proc, X_raw, y_raw, y_presence, intensity_model):
        # Pairwise-eligible rows: at least 2 materials present.
        # Replaces the old all-present complete-case filter so smaller groups
        # (S, W) are no longer starved of residual rows.
        usable = y_presence.sum(axis=1) >= 2
        if usable.sum() < len(Y_COLS):
            raise ValueError(f"Need >= {len(Y_COLS)} pairwise-usable rows; got {usable.sum()}.")

        mu_log = intensity_model.predict_log(X_proc)

        # Log-residuals for usable rows; NaN where material is absent
        p_sub  = y_presence[usable]
        y_sub  = y_raw[usable].astype(np.float64)
        y_log  = np.where(p_sub, np.log(np.where(p_sub, y_sub + LOG_EPS, 1.0)), np.nan)
        residuals = y_log - mu_log[usable]   # (n_usable, M); NaN for absent materials

        groups = build_group_keys(X_raw.loc[usable].reset_index(drop=True), self.group_cols)

        # ── Diagnostics ────────────────────────────────────────────────────────
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
        for m1 in range(len(Y_COLS)):
            for m2 in range(m1, len(Y_COLS)):
                both = p_sub[:, m1] & p_sub[:, m2]
                print(f"  ({Y_COLS[m1]:<10}, {Y_COLS[m2]:<10})  {int(both.sum()):>7}")

        # ── Global covariance (pairwise, all usable rows) ──────────────────────
        global_raw = self._pairwise_cov(residuals, p_sub)
        self.global_cov_ = self._regularise_cov(global_raw)

        # ── Per-group covariances ──────────────────────────────────────────────
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
                print(f"  {g:<10}  {n:>6}  {'—':>9}  {f'DROPPED':>10}")

        print(f"\n  Retained groups: {sorted(self.group_covs_.keys())}")
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
        self.stage2 = MaterialIntensityModel(**xgb_kw)
        self.joint = JointDistributionModel(
            group_cols=group_cols,
            min_group_size=min_group_size,
            reg_eps=reg_eps,
            cov_shrink=0.4,
        )

    def fit(self, X_proc, X_raw, y_raw, y_mask):
        # y_presence: reported AND positive — the true "material exists" signal
        y_presence = y_mask & (y_raw > 0)
        self.stage1.fit(X_proc, y_presence)
        self.stage2.fit(X_proc, y_raw, y_presence)
        self.joint.fit(X_proc, X_raw, y_raw, y_presence, self.stage2)

        # Per-material empirical intensity bounds (p5/p95) for post-sampling clamp.
        # Computed on training presence rows only; used in sample_query() to prevent
        # out-of-distribution intensity samples without truncating realistic extremes.
        self.intensity_bounds_ = []
        for m in range(len(Y_COLS)):
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
        proba = self.stage1.predict_proba(X_proc)
        intervals = self.stage2.predict_intervals(X_proc, alpha=alpha)
        for m, mat in enumerate(Y_COLS):
            intervals[mat]["p_presence"] = proba[:, m]
        return intervals

    def sample_query(
        self,
        X_proc,
        groups,
        n_samples=1000,
        temperature=2.5,
        random_state=None,
    ):
        """Sample from the full joint predictive distribution.

        Stage 1 — Temperature-scaled presence: divides chain logits by
                  `temperature` before Bernoulli draw.  temperature > 1
                  pushes probabilities toward 0.5, increasing combination
                  diversity and reducing over-saturation.
        Stage 2 — Quantile XGBoost sampling: recovers mu (p50) and sigma
                  (from p05/p95 spread) from the quantile model, then draws
                  from N(mu, sigma²) in log-space.
        Stage 3 — Active-material joint residual: applies the group
                  covariance only to the materials that are present in
                  each draw, avoiding spurious cross-material correlation
                  for absent materials.

        Parameters
        ----------
        X_proc      : np.ndarray (n_rows, n_features) preprocessed features
        groups      : np.ndarray (n_rows,) group labels from build_group_keys
        n_samples   : int   draws per query row
        temperature : float temperature for presence logit scaling (default 2.5)
        random_state: int or None

        Returns
        -------
        samples  : np.ndarray (n_rows, n_samples, n_materials)  kg/m²
        presence : np.ndarray (n_rows, n_samples, n_materials)  bool
        """
        rng    = np.random.default_rng(random_state)
        n_rows = X_proc.shape[0]
        M      = len(Y_COLS)

        # Stage 1: chain-sampled presence — preserves conditional co-occurrence structure
        all_presence = self.stage1.sample_presence(
            X_proc, n_samples=n_samples, temperature=temperature, random_state=rng,
            primary_codes=groups,
        )                                                            # (n_rows, n_samples, M)

        all_samples = np.zeros((n_rows, n_samples, M), dtype=np.float64)

        for i in range(n_rows):
            Xi    = X_proc[i : i + 1]                               # (1, n_feat)
            Sigma = self.joint.get_cov(groups[i])                   # (M, M)
            Z     = all_presence[i]                                  # (n_samples, M) bool


            for s in range(n_samples):
                y_log    = np.zeros(M)
                sigma_sq = np.zeros(M)   # Stage 2 per-material variance (for lognormal correction)

                for m, material in enumerate(Y_COLS):
                    if not Z[s, m]:
                        continue
                    qxgb = self.stage2.models_.get(material)
                    if qxgb is None:
                        continue

                    # Stage 2: sample from quantile-inferred N(mu, sigma²)
                    pq_log = qxgb.predict_q_log(Xi)                 # (1, 3)
                    mu     = float(pq_log[0, 1])
                    sigma  = max(float(pq_log[0, 2] - pq_log[0, 0]) / (2 * Z_QUANT_95), 1e-6)
                    y_log[m]    = rng.normal(mu, sigma)
                    sigma_sq[m] = sigma * sigma

                # Stage 3: joint residual with damped cross-covariance.
                # Off-diagonal entries of Sigma are scaled by CROSS_DAMP to separate
                # per-material intensity variance (diagonal, preserved) from cross-material
                # mean coupling (off-diagonal, reduced).  Prevents a high-intensity concrete
                # draw from inflating correlated materials (wood, brick) through the joint
                # residual — presence dependency is already handled by Stage 1 chain sampling.
                active = Z[s]
                if active.sum() >= 2:
                    idx      = np.where(active)[0]
                    Sig_sub  = Sigma[np.ix_(idx, idx)]
                    Sig_damp = CROSS_DAMP * Sig_sub
                    diag_pos = np.arange(len(idx))
                    Sig_damp[diag_pos, diag_pos] = Sig_sub[diag_pos, diag_pos]
                    eps_joint = rng.multivariate_normal(
                        np.zeros(len(idx)),
                        Sig_damp,
                    )
                    eps_joint *= RESIDUAL_SCALE
                    y_log[idx] += eps_joint

                # Lognormal mean correction: removes upward bias from E[exp(X)] = exp(mu + sigma²/2).
                # Diagonal of Sig_damp equals diagonal of Sigma, so sigma_eff² is unchanged.
                # sigma_eff² = sigma_stage2² + RESIDUAL_SCALE² * Sigma[m,m]
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
        n_samples=2000,
        temperature=2.5,
        random_state=SEED,
    ):
        """Post-hoc calibration of joint sampling thresholds.

        Fits one IsotonicRegression per material that maps the empirical
        per-row sampled presence frequency (after full chain coupling and
        temperature scaling) to the true binary presence on held-out
        validation data.  Calibrators are stored in
        stage1.presence_calibrators_ and applied automatically inside
        sample_presence() at inference time.

        The calibration target is the joint-sampled frequency, NOT the
        raw Stage 1 classifier probability, so it corrects for the
        over-saturation introduced by chain dependency coupling.

        Parameters
        ----------
        X_val          : np.ndarray (n_val, n_features)  preprocessed
        y_val_presence : np.ndarray (n_val, M) bool
                         true presence = observed AND positive
                         (same mask used in fit())
        n_samples      : int   draws per val row; more → stable frequency
                         estimates (default 2000)
        temperature    : float same value used in sample_query()
        random_state   : int or None

        Returns
        -------
        self
        """
        all_pres = self.stage1.sample_presence(
            X_val,
            n_samples=n_samples,
            temperature=temperature,
            random_state=random_state,
        )                                       # (n_val, n_samples, M)

        sampled_freq = all_pres.mean(axis=1)   # (n_val, M) — per-row empirical rate

        for m, material in enumerate(Y_COLS):
            x_cal = sampled_freq[:, m]                    # (n_val,)
            y_cal = y_val_presence[:, m].astype(float)    # (n_val,) 0/1
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x_cal, y_cal)
            self.stage1.presence_calibrators_[material] = iso

        return self

    def evaluate_sampling_realism(
        self, X_proc, groups,
        y_train_raw, y_train_mask, X_train_raw=None,
        n_samples=1000, random_state=None,
    ):
        """Print sampling realism diagnostics for each query row.

        Runs sample_query then reports five diagnostics per row:

        A  Presence frequency per material across samples.
        B  Co-occurrence matrix — fraction of samples where both present.
        C  Sampled intensity distribution: mean / median / p5 / p95.
        D  Diversity score — number of unique material combination patterns.
        E  Comparison to nearest real buildings (same group in training data).

        Parameters
        ----------
        X_proc       : np.ndarray (n_rows, n_features)
        groups       : np.ndarray (n_rows,) from build_group_keys
        y_train_raw  : np.ndarray (n_train, n_materials)  training intensities
        y_train_mask : np.ndarray (n_train, n_materials)  bool, observed (notna)
        X_train_raw  : pd.DataFrame (n_train, ...)  raw training features for
                       group matching; if None, all training rows are used.
        n_samples    : int
        random_state : int or None
        """
        samples, presence = self.sample_query(
            X_proc, groups, n_samples=n_samples, random_state=random_state
        )

        train_groups = (build_group_keys(X_train_raw)
                        if X_train_raw is not None else None)
        M   = len(Y_COLS)
        SEP = "=" * 60

        for i in range(X_proc.shape[0]):
            samp = samples[i]    # (n_samples, M)
            pres = presence[i]   # (n_samples, M) bool
            grp  = groups[i]

            print(f"\n{SEP}")
            print(f"Query row {i}  |  group: {grp}  |  n_samples={n_samples}")
            print(SEP)

            # ── A. Presence frequency ─────────────────────────────────────
            print("\nA. Presence frequency per material:")
            for m, mat in enumerate(Y_COLS):
                freq = float(pres[:, m].mean())
                bar  = "#" * int(round(freq * 20))
                print(f"   {mat:<12}  {freq:.3f}  |{bar:<20}|")

            # ── B. Co-occurrence matrix ───────────────────────────────────
            print("\nB. Co-occurrence matrix  (fraction of samples):")
            print("             " + "  ".join(f"{mat[:5]:>5}" for mat in Y_COLS))
            for m1, mat1 in enumerate(Y_COLS):
                row = f"  {mat1:<12}"
                for m2 in range(M):
                    v = float((pres[:, m1] & pres[:, m2]).mean())
                    row += f"  {v:.2f}"
                print(row)

            # ── C. Sampled intensity distributions ───────────────────────
            print("\nC. Sampled intensity distribution  (kg/m², presence rows only):")
            print(f"   {'Material':<12}  {'n_pres':>6}  {'mean':>8}  "
                  f"{'median':>8}  {'p5':>8}  {'p95':>8}")
            print("   " + "-" * 58)
            for m, mat in enumerate(Y_COLS):
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

            # ── D. Diversity score ────────────────────────────────────────
            combos, cnts = np.unique(
                pres.astype(np.int8), axis=0, return_counts=True
            )
            n_unique = len(combos)
            print(f"\nD. Diversity: {n_unique} unique material combinations"
                  f" / {n_samples} samples")
            print("   Top 5 combinations:")
            for idx in np.argsort(-cnts)[:5]:
                labels    = [Y_COLS[m] for m in range(M) if combos[idx, m]]
                label_str = ", ".join(labels) if labels else "(no materials)"
                print(f"     {label_str:<50}  {cnts[idx]:4d}  ({cnts[idx]/n_samples:.1%})")

            # ── E. Reference comparison ───────────────────────────────────
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
            for m, mat in enumerate(Y_COLS):
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
        """Compute scalar validation metrics comparing samples to reference data.

        Parameters
        ----------
        samples       : np.ndarray (n_rows, n_samples, M)  kg/m² from sample_query()
        presence      : np.ndarray (n_rows, n_samples, M)  bool from sample_query()
        y_ref         : np.ndarray (n_ref, M)              reference intensities
        y_ref_mask    : np.ndarray (n_ref, M)  bool        observed AND positive
        groups_ref    : np.ndarray (n_ref,) or None        group keys for y_ref rows
        query_groups  : np.ndarray (n_rows,) or None       group keys for query rows

        Returns
        -------
        dict with per-material entries, each containing:
          pres_freq_err  absolute |sampled_freq − true_freq|
          kl_div         KL divergence KL(true_intensity || sampled_intensity)
          wasserstein1   Wasserstein-1 (earth mover's) distance on log-space intensities
        """
        from scipy.stats import wasserstein_distance

        M = len(Y_COLS)
        results = {}

        for m, mat in enumerate(Y_COLS):
            # ── Presence frequency error ───────────────────────────────────────
            samp_freq = float(presence[:, :, m].mean())
            true_freq = float(y_ref_mask[:, m].mean())
            pres_err  = abs(samp_freq - true_freq)

            # ── Intensity distributions (log-space, presence rows only) ───────
            samp_vals = samples[:, :, m][presence[:, :, m]]   # flattened
            ref_rows  = y_ref_mask[:, m]
            ref_vals  = y_ref[ref_rows, m]

            if len(samp_vals) < 2 or len(ref_vals) < 2:
                results[mat] = dict(pres_freq_err=pres_err, kl_div=np.nan, wasserstein1=np.nan)
                continue

            samp_log = np.log(samp_vals + LOG_EPS)
            ref_log  = np.log(ref_vals  + LOG_EPS)

            # Wasserstein-1 on log-space intensities
            w1 = float(wasserstein_distance(ref_log, samp_log))

            # KL divergence via histogram approximation (true || sampled)
            edges      = np.linspace(
                min(ref_log.min(), samp_log.min()),
                max(ref_log.max(), samp_log.max()),
                31,
            )
            p_ref,  _  = np.histogram(ref_log,  bins=edges, density=True)
            p_samp, _  = np.histogram(samp_log, bins=edges, density=True)
            # Add small epsilon to avoid log(0); re-normalise
            _eps       = 1e-10
            p_ref      = p_ref  + _eps;  p_ref  /= p_ref.sum()
            p_samp     = p_samp + _eps;  p_samp /= p_samp.sum()
            kl         = float(np.sum(p_ref * np.log(p_ref / p_samp)))

            results[mat] = dict(pres_freq_err=pres_err, kl_div=kl, wasserstein1=w1)

        return results

    def print_sample_metrics(
        self, samples, presence, y_ref, y_ref_mask, groups_ref=None, query_groups=None,
    ):
        """Print evaluate_samples() as a formatted table."""
        metrics = self.evaluate_samples(
            samples, presence, y_ref, y_ref_mask, groups_ref, query_groups,
        )
        print(f"\n{'Material':<12}  {'Pres |err|':>10}  {'KL div':>8}  {'Wass-1':>8}")
        print("-" * 44)
        for mat in Y_COLS:
            r  = metrics[mat]
            kl = f"{r['kl_div']:>8.4f}" if not np.isnan(r["kl_div"]) else f"{'—':>8}"
            w1 = f"{r['wasserstein1']:>8.4f}" if not np.isnan(r["wasserstein1"]) else f"{'—':>8}"
            print(f"{mat:<12}  {r['pres_freq_err']:>10.4f}  {kl}  {w1}")

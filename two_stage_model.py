import numpy as np
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier, XGBRegressor

SEED = 42
LOG_EPS = 1e-6
Y_COLS = ["Concrete", "Glass", "Steel", "Wood", "Brick"]
GROUP_COLS = ["Primary Code"]


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

    Trains one XGBClassifier per material in prevalence-descending order.
    Each classifier conditions on all previously fitted materials, capturing
    co-occurrence dependencies that independent Bernoulli sampling misses.

    fit / predict_proba keep the same signatures as before.
    calibrate()       fits per-material IsotonicRegression on a held-out set.
    sample_presence() draws coherent joint combinations via ancestral sampling,
                      applying calibration at each chain step if fitted.
    evaluate_calibration() prints reliability diagnostics and returns plot data.
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
        self.calibrators_ = {}
        self.chain_order_ = list(range(len(Y_COLS)))

    @staticmethod
    def _chain_X(X, ctx_cols):
        """Append accumulated chain columns to base features."""
        if not ctx_cols:
            return X
        return np.hstack([X, np.column_stack(ctx_cols)])

    def fit(self, X, y_presence):
        """Fit classifier chain in descending binary-entropy order.

        Each XGBClassifier is wrapped with CalibratedClassifierCV so that
        predict_proba() outputs calibrated probabilities directly.
        Isotonic regression is used when n_samples >= 100 (enough data for
        monotone fit); Platt scaling (sigmoid) is used otherwise.
        """
        self.models_ = {}
        self.trivial_proba_ = {}
        self.calibrators_ = {}   # reset: calibration is now baked into models_
        p = y_presence.mean(axis=0).clip(1e-9, 1 - 1e-9)
        entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
        self.chain_order_ = list(np.argsort(-entropy))
        n = X.shape[0]
        cal_method = "isotonic" if n >= 100 else "sigmoid"
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

    def calibrate(self, X_val, y_val_presence):
        """Fit per-material isotonic calibrators on held-out presence labels.

        For each material in chain order, computes the raw conditional
        probability P(z_m=1 | X_val, true_prev_labels) then fits an
        IsotonicRegression to map those raw probabilities to calibrated ones.
        Calibrators are applied automatically by predict_proba and
        sample_presence after this call.

        Parameters
        ----------
        X_val          : np.ndarray (n_val, n_features) preprocessed features
        y_val_presence : np.ndarray (n_val, n_materials) bool  true presence
        """
        self.calibrators_ = {}
        ctx = []   # growing true-label context (n_val,) per material
        for m in self.chain_order_:
            material = Y_COLS[m]
            obs = y_val_presence[:, m].astype(int)
            if self.models_.get(material) is None:
                self.calibrators_[material] = None
            else:
                p_raw = self.models_[material].predict_proba(
                    self._chain_X(X_val, ctx)
                )[:, 1]
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(p_raw, obs)
                self.calibrators_[material] = cal
            ctx.append(y_val_presence[:, m].astype(np.float64))
        return self

    def predict_proba(self, X):
        """Marginal probabilities via greedy chain with isotonic calibration."""
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
            cal = self.calibrators_.get(material)
            if cal is not None:
                p_m = cal.predict(p_m)
            proba[:, m] = p_m
            ctx.append((p_m > 0.5).astype(np.float64))
        return proba

    def sample_presence(self, X, n_samples=1000, random_state=None):
        """Draw coherent material combinations via ancestral chain sampling.

        Sequentially samples each material conditioned on previously sampled
        materials.  Applies isotonic calibration at each chain step if
        calibrate() has been called.

        Parameters
        ----------
        X           : np.ndarray (n_rows, n_features) preprocessed features
        n_samples   : int  draws per query row
        random_state: int, Generator, or None

        Returns
        -------
        np.ndarray (n_rows, n_samples, n_materials) bool
        """
        rng    = np.random.default_rng(random_state)
        n_rows = X.shape[0]
        M      = len(Y_COLS)
        out    = np.zeros((n_rows, n_samples, M), dtype=bool)

        for i in range(n_rows):
            X_rep = np.tile(X[i:i+1], (n_samples, 1))   # (n_samples, n_feat)
            ctx   = []                                    # (n_samples,) cols
            pres  = np.zeros((n_samples, M), dtype=bool)

            for m in self.chain_order_:
                material = Y_COLS[m]
                if self.models_.get(material) is None:
                    p_m = np.full(n_samples,
                                  self.trivial_proba_.get(material, 0.0))
                else:
                    p_m = self.models_[material].predict_proba(
                        self._chain_X(X_rep, ctx)
                    )[:, 1]                               # (n_samples,)
                cal = self.calibrators_.get(material)
                if cal is not None:
                    p_m = cal.predict(p_m)
                z = rng.random(n_samples) < p_m
                pres[:, m] = z
                ctx.append(z.astype(np.float64))

            out[i] = pres

        return out

    def evaluate_calibration(self, X, y_presence, n_bins=10):
        """Reliability diagnostics for Stage 1 presence probabilities.

        Uses the current predict_proba output (calibrated if calibrate() has
        been called, raw otherwise) to compute per-material metrics and
        reliability curve data.

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
        proba     = self.predict_proba(X)
        cal_state = ("calibrated"
                     if any(v is not None for v in self.calibrators_.values())
                     else "raw")

        print(f"  {'Material':<12}  {'Brier':>8}  {'Mean pred':>10}  "
              f"{'Obs freq':>10}  {'Bias':>8}  State: {cal_state}")
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

        # Store GMM and label mapping so held-out y can be assigned to regimes.
        self.gm_ = gm
        self.orig_to_compact_ = {int(orig): compact for compact, orig in enumerate(unique)}
        self.train_labels_ = raw_labels.copy()
        self.train_y_log_ = y_log.copy()

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

    def _regime_labels_from_y(self, y_log):
        """Assign log-target values to compact regime labels using the fitted GMM."""
        gm_preds = self.gm_.predict(y_log.reshape(-1, 1))
        return np.array([self.orig_to_compact_.get(int(p), -1) for p in gm_preds])

    def evaluate_gating(self, X, y_log):
        """Print gating classifier diagnostics against GMM-derived regime labels."""
        if self.K_ == 1:
            print("  Gating trivial: K=1 (single regime, no classifier needed).")
            return

        true_labels = self._regime_labels_from_y(y_log)
        valid = true_labels >= 0
        if valid.sum() == 0:
            print("  No valid regime labels to evaluate (all GMM components unseen).")
            return

        X_ev, y_true = X[valid], true_labels[valid]
        y_pred = self.gate_.predict(X_ev)

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)

        # Reorder predict_proba columns to match compact regime indices.
        proba_raw = self.gate_.predict_proba(X_ev)
        gate_proba = np.zeros((X_ev.shape[0], self.K_))
        for col, cls in enumerate(self.gate_.classes_):
            gate_proba[:, int(cls)] = proba_raw[:, col]
        mean_conf = float(gate_proba.max(axis=1).mean())

        print(f"  Gating accuracy : {acc:.3f}  (K={self.K_} regimes, n={valid.sum()})")
        print(f"  Mean max gating confidence : {mean_conf:.3f}")
        print("  Confusion matrix (rows=true regime, cols=predicted regime):")
        print(cm)
        print("  Per-class precision / recall / F1:")
        print(report)

        # Interpretation
        if acc < 0.50:
            if mean_conf > 0.70:
                print(
                    "  [Interpretation] High confidence + low accuracy: the gating model is\n"
                    "  decisive but systematically wrong. Likely cause — regime overlap or\n"
                    "  bad GMM clustering: the GMM split y into regimes that are not\n"
                    "  separable in X-space (features carry no signal about which regime a\n"
                    "  sample belongs to). Consider reducing K or using a different\n"
                    "  clustering criterion."
                )
            else:
                print(
                    "  [Interpretation] Low accuracy + low confidence: regime assignment\n"
                    "  is not learnable from X. The features do not predict which regime a\n"
                    "  sample falls into, so MoE gating is near-random. The model falls back\n"
                    "  to a uniform mixture, which provides wider intervals but no regime\n"
                    "  specialisation benefit."
                )
        elif acc < 0.70:
            if mean_conf > 0.70:
                print(
                    "  [Interpretation] Moderate accuracy + high confidence: gating is\n"
                    "  decisive but makes systematic errors on some regimes. Check per-class\n"
                    "  recall — a minority regime with low recall is being swallowed by the\n"
                    "  majority regime."
                )
            else:
                print(
                    "  [Interpretation] Moderate accuracy + low confidence: gating is\n"
                    "  uncertain. MoE averages over regimes with similar weights, which still\n"
                    "  inflates prediction intervals appropriately but provides limited\n"
                    "  expert specialisation."
                )
        else:
            print(
                "  [Interpretation] Good gating accuracy — regime assignment is learnable\n"
                "  from X. MoE gating is functioning as intended."
            )

    def evaluate_expert_diversity(self, X):
        """Print expert specialisation diagnostics on held-out X.

        Covers four things:
          1. Regime sample counts from training.
          2. Target mean/std per regime (log-space) from training.
          3. Pairwise correlation between expert predictions on X.
          4. Mean absolute difference between expert predictions on X.
        """
        K = self.K_

        # 1. Regime sample counts
        counts = np.bincount(self.train_labels_, minlength=K)
        print(f"  Regime sample counts (training) : { {k: int(counts[k]) for k in range(K)} }")

        # 2. Target stats per regime (log-space, training data)
        print("  Target log-space stats per regime (training):")
        for k in range(K):
            mask = self.train_labels_ == k
            vals = self.train_y_log_[mask]
            print(f"    Regime {k}: n={int(mask.sum()):4d}  mean={vals.mean():.3f}  std={vals.std():.3f}")

        if K == 1:
            print("  Only 1 regime — no pairwise expert comparison.")
            return

        # 3 & 4. Expert predictions on held-out X
        preds = np.stack([self.experts_[k].predict(X) for k in range(K)], axis=0)  # (K, n)

        corr = np.corrcoef(preds)  # (K, K)
        print("  Pairwise correlation between expert predictions (log-space):")
        header = "         " + "  ".join(f"Exp{k}" for k in range(K))
        print(header)
        for k1 in range(K):
            row = f"    Exp{k1}  " + "  ".join(f"{corr[k1, k2]:+.3f}" for k2 in range(K))
            print(row)

        print("  Mean |expert_i − expert_j| on held-out X (log-space):")
        for k1 in range(K):
            for k2 in range(k1 + 1, K):
                diff = float(np.mean(np.abs(preds[k1] - preds[k2])))
                print(f"    |Exp{k1} − Exp{k2}| = {diff:.4f}")

        # Collapsed flag: high correlation + small diff ⇒ experts not specialised
        pairs = [(k1, k2) for k1 in range(K) for k2 in range(k1 + 1, K)]
        mean_corr = float(np.mean([corr[k1, k2] for k1, k2 in pairs]))
        mean_diff = float(np.mean([np.mean(np.abs(preds[k1] - preds[k2])) for k1, k2 in pairs]))
        if mean_corr > 0.95 and mean_diff < 0.10:
            print(
                "  [Interpretation] Experts are nearly identical (corr > 0.95, mean diff < 0.10).\n"
                "  The MoE is not benefiting from specialisation — experts have collapsed.\n"
                "  Possible causes: regime boundaries too similar, K too large, or training\n"
                "  data insufficient to differentiate expert behaviours."
            )
        elif mean_corr > 0.85:
            print(
                "  [Interpretation] Experts are moderately correlated (corr > 0.85).\n"
                "  Some specialisation present but limited. Check whether regime means\n"
                "  (above) differ substantially — if not, consider reducing K."
            )
        else:
            print(
                "  [Interpretation] Experts show meaningful diversity — regime specialisation\n"
                "  is working as intended."
            )

    def crps_gaussian(self, X, y_log):
        """Analytical Gaussian CRPS for the mixture predictive distribution (log-space).

        Uses the law-of-total-variance mixture mu/sigma (same as predict_quantiles),
        then applies: CRPS = sigma*(z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi))
        where z = (y - mu) / sigma.  Returns mean CRPS over samples.
        """
        K = self.K_
        w = self._weights(X)                                               # (n, K)
        mu = np.stack([self.experts_[k].predict(X) for k in range(K)], axis=1)
        sigma_k = self.expert_sigmas_                                      # (K,)

        mu_mix = (w * mu).sum(axis=1)                                      # (n,)
        within_var = (w * sigma_k[np.newaxis, :] ** 2).sum(axis=1)
        between_var = (w * (mu - mu_mix[:, np.newaxis]) ** 2).sum(axis=1)
        sigma_mix = np.sqrt(np.maximum(within_var + between_var, 1e-12))   # (n,)

        z = (y_log - mu_mix) / sigma_mix
        crps = sigma_mix * (
            z * (2.0 * stats.norm.cdf(z) - 1.0)
            + 2.0 * stats.norm.pdf(z)
            - 1.0 / np.sqrt(np.pi)
        )
        return float(np.mean(crps))

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

    def evaluate_gating(self, X, y):
        """Print gating diagnostics per material on held-out (val or test) data.

        Parameters
        ----------
        X : np.ndarray  preprocessed feature matrix (n_samples, n_features)
        y : np.ndarray or DataFrame  raw targets (n_samples, n_materials), NaN for missing
        """
        if hasattr(y, "to_numpy"):
            y = y.to_numpy(dtype=np.float64)

        for m, material in enumerate(Y_COLS):
            print(f"\nMaterial: {material}")
            moe = self.models_.get(material)
            if moe is None:
                print("  No model fitted (insufficient training data).")
                continue
            obs = (~np.isnan(y[:, m])) & (y[:, m] > 0)
            if obs.sum() < 2:
                print("  Insufficient observed presence rows for evaluation.")
                continue
            y_log = np.log(y[obs, m] + LOG_EPS)
            moe.evaluate_gating(X[obs], y_log)

    def evaluate_expert_diversity(self, X, y):
        """Print expert diversity diagnostics per material on held-out data.

        Parameters
        ----------
        X : np.ndarray  preprocessed feature matrix (n_samples, n_features)
        y : np.ndarray or DataFrame  raw targets (n_samples, n_materials), NaN for missing
        """
        if hasattr(y, "to_numpy"):
            y = y.to_numpy(dtype=np.float64)

        for m, material in enumerate(Y_COLS):
            print(f"\nMaterial: {material}")
            moe = self.models_.get(material)
            if moe is None:
                print("  No model fitted (insufficient training data).")
                continue
            obs = (~np.isnan(y[:, m])) & (y[:, m] > 0)
            if obs.sum() < 2:
                print("  Insufficient observed presence rows for evaluation.")
                continue
            moe.evaluate_expert_diversity(X[obs])

    def evaluate_crps(self, X, y):
        """Print mean CRPS per material on held-out data.

        CRPS (Continuous Ranked Probability Score) evaluates the full predictive
        distribution, not just the point forecast.  Lower is better; a perfectly
        calibrated Gaussian achieves CRPS = sigma * (1/sqrt(pi) - 1) ≈ −0.43*sigma.
        Reported in log-space so values are scale-comparable across materials.

        Parameters
        ----------
        X : np.ndarray  preprocessed feature matrix (n_samples, n_features)
        y : np.ndarray or DataFrame  raw targets (n_samples, n_materials), NaN for missing
        """
        if hasattr(y, "to_numpy"):
            y = y.to_numpy(dtype=np.float64)

        print(f"  {'Material':<12}  {'n_obs':>6}  {'Mean CRPS (log-space)':>22}")
        print("  " + "-" * 44)
        for m, material in enumerate(Y_COLS):
            moe = self.models_.get(material)
            if moe is None:
                print(f"  {material:<12}  {'—':>6}  {'no model':>22}")
                continue
            obs = (~np.isnan(y[:, m])) & (y[:, m] > 0)
            if obs.sum() < 2:
                print(f"  {material:<12}  {'—':>6}  {'too few rows':>22}")
                continue
            y_log = np.log(y[obs, m] + LOG_EPS)
            crps = moe.crps_gaussian(X[obs], y_log)
            print(f"  {material:<12}  {int(obs.sum()):>6}  {crps:>22.4f}")

    def evaluate_calibration(self, X, y, levels=None):
        """Empirical coverage at each nominal level per material.

        For each nominal coverage level (default 10%–90% in steps of 10%),
        computes the fraction of held-out presence rows whose true value falls
        inside the symmetric predictive interval.  Returns a dict suitable for
        calibration plotting; also prints a table.

        Parameters
        ----------
        X      : np.ndarray  preprocessed feature matrix (n_samples, n_features)
        y      : np.ndarray or DataFrame  raw targets, NaN for missing
        levels : array-like of floats in (0, 1), default np.linspace(0.10, 0.90, 9)

        Returns
        -------
        dict[material, np.ndarray]  empirical coverages aligned with levels
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
            moe = self.models_.get(material)
            if moe is None:
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
                p_lo, _, p_hi = moe.predict_quantiles(X_obs, q_lo=alpha / 2.0, q_hi=1.0 - alpha / 2.0)
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

    def calibrate_stage1(self, X_val, y_val_raw, y_val_mask):
        """Calibrate Stage 1 presence probabilities on validation data.

        Derives y_val_presence = y_val_mask & (y_val_raw > 0) and delegates
        to stage1.calibrate().  Call once after fit(), before sampling.

        Parameters
        ----------
        X_val       : np.ndarray (n_val, n_features) preprocessed features
        y_val_raw   : np.ndarray (n_val, n_materials) raw intensities
        y_val_mask  : np.ndarray (n_val, n_materials) bool, observed (notna)
        """
        self.stage1.calibrate(X_val, y_val_mask & (y_val_raw > 0))
        return self

    def sample_query(
        self,
        X_proc,
        groups,
        n_samples=1000,
        temperature=1.5,
        random_state=None,
    ):
        """Sample from the full joint predictive distribution.

        Stage 1 — Temperature-scaled presence: divides chain logits by
                  `temperature` before Bernoulli draw.  temperature > 1
                  pushes probabilities toward 0.5, increasing combination
                  diversity and reducing over-saturation.
        Stage 2 — Hard expert routing: samples a discrete regime k from
                  the gating distribution, then draws from that expert's
                  N(mu_k, sigma_k²) rather than the mixture mean.
        Stage 3 — Active-material joint residual: applies the group
                  covariance only to the materials that are present in
                  each draw, avoiding spurious cross-material correlation
                  for absent materials.

        Parameters
        ----------
        X_proc      : np.ndarray (n_rows, n_features) preprocessed features
        groups      : np.ndarray (n_rows,) group labels from build_group_keys
        n_samples   : int   draws per query row
        temperature : float temperature for presence logit scaling (default 1.5)
        random_state: int or None

        Returns
        -------
        samples  : np.ndarray (n_rows, n_samples, n_materials)  kg/m²
        presence : np.ndarray (n_rows, n_samples, n_materials)  bool
        """
        rng    = np.random.default_rng(random_state)
        n_rows = X_proc.shape[0]
        M      = len(Y_COLS)
        _eps   = 1e-9

        # Stage 1: temperature-scaled presence probabilities
        p_raw  = self.stage1.predict_proba(X_proc)                  # (n_rows, M)
        logit_p = np.log(p_raw.clip(_eps, 1 - _eps) /
                         (1 - p_raw.clip(_eps, 1 - _eps)))
        p_pres  = 1.0 / (1.0 + np.exp(-logit_p / temperature))     # (n_rows, M)

        all_samples  = np.zeros((n_rows, n_samples, M), dtype=np.float64)
        all_presence = np.zeros((n_rows, n_samples, M), dtype=bool)

        for i in range(n_rows):
            Xi    = X_proc[i : i + 1]                               # (1, n_feat)
            Sigma = self.joint.get_cov(groups[i])                   # (M, M)

            # Presence draws for all samples at once
            Z = rng.random((n_samples, M)) < p_pres[i]             # (n_samples, M)
            all_presence[i] = Z

            for s in range(n_samples):
                y_log = np.zeros(M)

                for m, material in enumerate(Y_COLS):
                    if not Z[s, m]:
                        continue
                    moe = self.stage2.models_.get(material)
                    if moe is None:
                        continue

                    # Stage 2: sample a discrete expert regime
                    if moe.K_ == 1:
                        k = 0
                    else:
                        gate_w = moe._weights(Xi)[0]                # (K,)
                        gate_w = gate_w / gate_w.sum()              # guard fp drift
                        k = int(rng.choice(moe.K_, p=gate_w))

                    mu    = float(moe.experts_[k].predict(Xi)[0])
                    sigma = float(moe.expert_sigmas_[k])
                    y_log[m] = rng.normal(mu, sigma)

                # Stage 3: joint residual for active materials only
                active = Z[s]
                if active.sum() >= 2:
                    idx       = np.where(active)[0]
                    eps_joint = rng.multivariate_normal(
                        np.zeros(len(idx)),
                        Sigma[np.ix_(idx, idx)],
                    )
                    y_log[idx] += eps_joint

                y = np.maximum(np.exp(y_log) - LOG_EPS, 0.0)
                y *= Z[s]
                all_samples[i, s] = y

        return all_samples, all_presence

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

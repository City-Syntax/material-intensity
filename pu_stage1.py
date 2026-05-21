"""
PU-learning Stage 1 — Elkan-Noto corrected material presence model.

Background
----------
The dataset is *positive-unlabeled* (PU):
  s = 1  →  confirmed presence   (observed AND quantity > 0)
  s = 0  →  UNLABELED            (either absent OR unobserved; NOT confirmed absent)

Elkan & Noto (2008) show that under the SCAR assumption
(labels are selected completely at random from true positives):

    P(y=1 | x) = P(s=1 | x) / c

where  c = P(s=1 | y=1)  is the constant labeling rate, estimated as

    c_hat = E[ P(s=1|x) | s=1 ]   (mean model score on confirmed positives)

and the class prior is

    π = P(y=1) ≈ E[ P(s=1|x) ] / c_hat   (across all rows)

Drop-in replacement
-------------------
`PUMaterialOccurrenceModel` has the same public interface as the original
`MaterialOccurrenceModel`:

    model.fit(X_proc, y_presence)            # y_presence = s labels
    model.predict_proba(X_proc)              # now returns P(y=1|x)
    model.sample_presence(...)               # draws from P(y=1|x)
    model.c_          # dict: material → c   (labeling rate)
    model.pi_         # dict: material → π   (class prior P(y=1))

Chain context design
--------------------
- During TRAINING  : chain context = binary s labels  (same as before)
- During INFERENCE (predict_proba / _predict_proba_order):
      context built from  P(s=1|x) > 0.5  (UNCORRECTED)
      so that probability inflation from ÷c does not cascade through the chain
- During SAMPLING  (sample_presence):
      p_chain drawn from  P(y=1|x)  (CORRECTED)
      context built from  sampled binary z  (no cascading issue)

After calibrate_sampling()
--------------------------
The post-hoc IsotonicRegression is fitted against s_val (observed labels),
so the *sampling* frequencies align to observed rates (correct for Stage 2
conditioning).  The `predict_proba()` output is PU-corrected and NOT
post-hoc-scaled, giving interpretable P(y=1|x) estimates.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier


# ── These must match the notebook globals ──────────────────────────────────────
BLEND_LAMBDA = 0.35
DROPOUT_RATE = 0.08
P_CLIP_LO    = 0.05
P_CLIP_HI    = 0.95
SEED         = 42
y_cols       = ["Concrete", "Glass", "Steel", "Wood", "Brick"]


# ══════════════════════════════════════════════════════════════════════════════
# Elkan-Noto PU wrapper
# ══════════════════════════════════════════════════════════════════════════════

class ElkanNotoPUWrapper:
    """
    Wraps CalibratedClassifierCV(XGBClassifier, sigmoid, cv=5) with the
    Elkan-Noto probability correction for PU data.

    Attributes
    ----------
    c_   : float   estimated labeling rate  P(s=1 | y=1)
    pi_  : float   estimated class prior    P(y=1)
    base_estimator_ : fitted CalibratedClassifierCV
    """

    def __init__(self, xgb_params: dict, cv: int = 5, min_c: float = 0.05):
        self.xgb_params = xgb_params
        self.cv         = cv
        self.min_c      = min_c

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, s: np.ndarray) -> "ElkanNotoPUWrapper":
        """
        Parameters
        ----------
        X : (n, d) preprocessed feature matrix
        s : (n,)  PU labels — 1 = confirmed positive, 0 = unlabeled
        """
        s = np.asarray(s, dtype=int)
        base = XGBClassifier(**self.xgb_params)
        cal  = CalibratedClassifierCV(base, method="sigmoid", cv=self.cv)
        cal.fit(X, s)
        self.base_estimator_ = cal

        # Elkan-Noto c-estimate: mean P(s=1|x) over confirmed positives
        pos_mask = s == 1
        n_pos    = int(pos_mask.sum())
        if n_pos >= 3:
            p_pos   = cal.predict_proba(X[pos_mask])[:, 1]
            self.c_ = float(np.clip(np.mean(p_pos), self.min_c, 1.0))
        else:
            # Degenerate: too few positives → treat as fully supervised
            self.c_ = 1.0

        # Class prior: π = mean(P(s=1|x)) / c  over ALL rows
        p_all    = cal.predict_proba(X)[:, 1]
        self.pi_ = float(np.clip(np.mean(p_all) / self.c_, 0.0, 1.0))

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (n, 2): [:, 1] = P(y=1|x) = clip(P(s=1|x) / c, 0, 1).
        Used for final output and sampling draws.
        """
        p_s  = self.base_estimator_.predict_proba(X)[:, 1]
        p_y1 = np.clip(p_s / self.c_, 0.0, 1.0)
        return np.column_stack([1.0 - p_y1, p_y1])

    def predict_proba_s(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (n, 2): [:, 1] = P(s=1|x) — UNCORRECTED.
        Used only for building chain context in predict_proba() /
        _predict_proba_order() to prevent probability inflation from
        cascading through the chain.
        """
        return self.base_estimator_.predict_proba(X)


# ══════════════════════════════════════════════════════════════════════════════
# PU-corrected Material Occurrence Model (drop-in Stage 1 replacement)
# ══════════════════════════════════════════════════════════════════════════════

class PUMaterialOccurrenceModel:
    """
    Classifier chain for joint material presence modeling — PU-learning variant.

    Identical public interface to MaterialOccurrenceModel.
    Each chain node is an ElkanNotoPUWrapper that outputs P(y=1|x) instead
    of the naive P(s=1|x).

    Extra attributes
    ----------------
    c_   : dict  material → estimated labeling rate c = P(s=1 | y=1)
    pi_  : dict  material → estimated class prior π = P(y=1)
    """

    def __init__(
        self,
        n_estimators    = 300,
        max_depth       = 4,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        reg_alpha       = 0.0,
        reg_lambda      = 1.0,
        random_state    = SEED,
    ):
        self.xgb_params = dict(
            n_estimators    = n_estimators,
            max_depth       = max_depth,
            learning_rate   = learning_rate,
            subsample       = subsample,
            colsample_bytree= colsample_bytree,
            reg_alpha       = reg_alpha,
            reg_lambda      = reg_lambda,
            random_state    = random_state,
            objective       = "binary:logistic",
            verbosity       = 0,
        )
        self.models_                   : dict = {}
        self.trivial_proba_            : dict = {}
        self.chain_order_              : list = list(range(len(y_cols)))
        self.presence_calibrators_     : dict = {}
        self.presence_group_calibrators_: dict = {}
        # PU-specific
        self.c_   : dict = {}   # labeling rate per material
        self.pi_  : dict = {}   # class prior per material

    # ------------------------------------------------------------------
    @staticmethod
    def _chain_X(X: np.ndarray, ctx_cols: list) -> np.ndarray:
        if not ctx_cols:
            return X
        return np.hstack([X, np.column_stack(ctx_cols)])

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y_presence: np.ndarray) -> "PUMaterialOccurrenceModel":
        """
        Fit classifier chain in descending binary-entropy order.

        Parameters
        ----------
        X          : (n, d) preprocessed feature matrix
        y_presence : (n, M) bool — s labels; True = confirmed present
        """
        self.models_        = {}
        self.trivial_proba_ = {}
        self.c_             = {}
        self.pi_            = {}

        p_vec   = y_presence.mean(axis=0).clip(1e-9, 1 - 1e-9)
        entropy = -p_vec * np.log(p_vec) - (1 - p_vec) * np.log(1 - p_vec)
        self.chain_order_ = list(np.argsort(-entropy))

        # ctx: binary s labels used as chain context during training
        ctx: list = []
        for m in self.chain_order_:
            material = y_cols[m]
            s = y_presence[:, m].astype(int)
            p = float(s.mean())

            if p == 0.0 or p == 1.0:
                self.trivial_proba_[material] = p
                self.models_[material]        = None
                self.c_[material]             = 1.0
                self.pi_[material]            = p
            else:
                wrapper = ElkanNotoPUWrapper(self.xgb_params, cv=5)
                wrapper.fit(self._chain_X(X, ctx), s)
                self.models_[material] = wrapper
                self.c_[material]      = wrapper.c_
                self.pi_[material]     = wrapper.pi_

            # Training context = raw binary s labels (same as original)
            ctx.append(y_presence[:, m].astype(np.float64))

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Marginal P(y=1|x) via greedy chain — Elkan-Noto corrected.

        Chain context is built from UNCORRECTED P(s=1|x) > 0.5 to
        prevent probability inflation from cascading.

        Returns
        -------
        (n, M) float  — P(y=1|x) for each material
        """
        n     = X.shape[0]
        proba = np.zeros((n, len(y_cols)), dtype=np.float64)
        ctx   : list = []

        for m in self.chain_order_:
            material = y_cols[m]
            clf      = self.models_.get(material)

            if clf is None:
                p_corr = np.full(n, self.trivial_proba_.get(material, 0.0))
                p_ctx  = p_corr  # trivial: same for context
            else:
                X_ch   = self._chain_X(X, ctx)
                p_corr = clf.predict_proba(X_ch)[:, 1]    # P(y=1|x)
                p_ctx  = clf.predict_proba_s(X_ch)[:, 1]  # P(s=1|x) for context

            proba[:, m] = p_corr
            ctx.append((p_ctx > 0.5).astype(np.float64))  # uncorrected hard label

        return proba

    # ------------------------------------------------------------------
    def _predict_proba_order(
        self,
        X        : np.ndarray,
        perm     : np.ndarray,
        train_pos: dict,
    ) -> np.ndarray:
        """Greedy chain predict_proba with an arbitrary material sampling order."""
        n         = X.shape[0]
        proba     = np.zeros((n, len(y_cols)))
        ctx_built : dict = {}

        for k in range(len(y_cols)):
            m        = perm[k]
            material = y_cols[m]
            t        = train_pos[m]

            ctx = [
                ctx_built.get(self.chain_order_[j], np.zeros(n))
                for j in range(t)
            ]
            clf = self.models_.get(material)

            if clf is None:
                p_corr = np.full(n, self.trivial_proba_.get(material, 0.0))
                p_ctx  = p_corr
            else:
                X_ch   = self._chain_X(X, ctx)
                p_corr = clf.predict_proba(X_ch)[:, 1]
                p_ctx  = clf.predict_proba_s(X_ch)[:, 1]

            proba[:, m]  = p_corr
            ctx_built[m] = (p_ctx > 0.5).astype(np.float64)

        return proba

    # ------------------------------------------------------------------
    def sample_presence(
        self,
        X            : np.ndarray,
        n_samples    : int   = 1000,
        temperature  : float = 1.0,
        random_state         = None,
        n_chain_orders: int  = 4,
        primary_codes        = None,
    ) -> np.ndarray:
        """
        Draw coherent material combinations via randomized ancestral chain sampling.

        Draws are from P(y=1|x) — the PU-corrected probability — which is the
        correct distribution for estimating true material presence.  After
        calibrate_sampling(), an IsotonicRegression maps sampled frequencies to
        observed (s_val) rates so Stage 2 conditioning remains valid.

        Returns
        -------
        np.ndarray (n_rows, n_samples, M) bool
        """
        _eps = 1e-9

        rng    = np.random.default_rng(random_state)
        n_rows = X.shape[0]
        M      = len(y_cols)
        out    = np.zeros((n_rows, n_samples, M), dtype=bool)

        train_pos = {m: pos for pos, m in enumerate(self.chain_order_)}

        for i in range(n_rows):
            Xi = X[i:i+1]

            # ── Blend anchor: order-averaged marginal ─────────────────────────
            p_avg = self.predict_proba(Xi)[0].copy()
            for _ in range(n_chain_orders - 1):
                perm   = rng.permutation(M)
                p_avg += self._predict_proba_order(Xi, perm, train_pos)[0]
            p_marginal = p_avg / n_chain_orders

            # ── Stochastic draws in n_chain_orders groups ─────────────────────
            pres  = np.zeros((n_samples, M), dtype=bool)
            edges = np.linspace(0, n_samples, n_chain_orders + 1, dtype=int)

            for g in range(n_chain_orders):
                start, end = int(edges[g]), int(edges[g + 1])
                n_g        = end - start
                X_rep      = np.tile(Xi, (n_g, 1))
                perm       = rng.permutation(M)
                pres_g     = np.zeros((n_g, M), dtype=bool)
                ctx_built  : dict = {}

                for k in range(M):
                    m        = perm[k]
                    material = y_cols[m]
                    t        = train_pos[m]

                    ctx = [
                        ctx_built.get(self.chain_order_[j], np.zeros(n_g))
                        for j in range(t)
                    ]
                    clf = self.models_.get(material)

                    if clf is None:
                        p_m = np.full(n_g, self.trivial_proba_.get(material, 0.0))
                    else:
                        X_ch    = self._chain_X(X_rep, ctx)
                        # Draw from PU-corrected P(y=1|x)
                        p_chain = clf.predict_proba(X_ch)[:, 1]
                        if temperature != 1.0:
                            logit = np.log(
                                p_chain.clip(_eps, 1 - _eps) /
                                (1 - p_chain.clip(_eps, 1 - _eps))
                            )
                            p_chain = 1.0 / (1.0 + np.exp(-logit / temperature))

                        p_base = np.full(n_g, float(p_marginal[m]))
                        p_m    = BLEND_LAMBDA * p_chain + (1.0 - BLEND_LAMBDA) * p_base

                    # Apply calibrator (maps sampled freq → s_val rates post-calibrate_sampling)
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


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic helper
# ══════════════════════════════════════════════════════════════════════════════

def pu_report(model: PUMaterialOccurrenceModel) -> None:
    """
    Print per-material PU correction statistics after fitting.

    Columns
    -------
    material : material name
    π (prior): estimated P(y=1) — true class prior
    c (label rate): estimated P(s=1|y=1) — fraction of true positives labeled
    s_rate   : raw label rate P(s=1) in training data
    correction ×: 1/c — factor applied to P(s=1|x) to recover P(y=1|x)
    """
    header = f"{'Material':<12}  {'π P(y=1)':>10}  {'c P(s|y=1)':>12}  "
    header += f"{'s_rate P(s=1)':>14}  {'correction ×':>14}"
    print(header)
    print("-" * len(header))
    for mat in y_cols:
        pi_  = model.pi_.get(mat, float("nan"))
        c_   = model.c_.get(mat, float("nan"))
        s_   = pi_ * c_ if (pi_ == pi_ and c_ == c_) else float("nan")
        corr = 1.0 / c_ if c_ and c_ > 0 else float("nan")
        print(
            f"{mat:<12}  {pi_:>10.4f}  {c_:>12.4f}  {s_:>14.4f}  {corr:>14.3f}×"
        )

#!/usr/bin/env python3
"""
generate_paper_figures.py
Regenerate all manuscript figures from the trained model + harmonised database.
Output: ../Nature_Scientific-Data_MI-database/figures/
"""

import sys, types, random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, gaussian_kde
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ── register model classes so joblib can deserialise ───────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_classes as _mc
_pm = types.ModuleType("prediction_model")
_pm.ObservationModel = _mc.ObservationModel
_pm.IntensityModel   = _mc.IntensityModel
_pm.FinalQueryModel  = _mc.FinalQueryModel
_mc.ObservationModel.__module__ = "prediction_model"
_mc.IntensityModel.__module__   = "prediction_model"
_mc.FinalQueryModel.__module__  = "prediction_model"
sys.modules.setdefault("prediction_model", _pm)

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR    = SCRIPT_DIR.parent / "ESSD_MI-database" / "figures"
DATA_FILE  = SCRIPT_DIR / "data" / "processed" / "Integrated_MI_database_add_Singapore.xlsx"
ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── constants (must match training) ────────────────────────────────────────
SEED = 42
MIN_OBSERVED_TARGETS = 2
y_cols        = ["Concrete", "Glass", "Steel", "Wood", "Brick"]
FIG_ORDER     = ["Concrete", "Steel", "Wood", "Brick", "Glass"]   # paper layout
PERIOD_BUCKETS = ["pre_1945", "1945_1970", "1970_1990", "1990_2010", "post_2010"]
X_cols = [
    "Construction period", "Construction period bucket", "Typology",
    "Primary Code", "Hybrid Structure", "Country", "Geo_macro",
]
FUNC_COL = "Function  Classification"   # note: two spaces in the Excel column name
CAL_ALPHA = 0.10

# ── colour palette ─────────────────────────────────────────────────────────
C_RES    = "#4C72B0"   # blue   – residential / Concrete
C_NR     = "#DD8452"   # orange – non-residential / Steel
C_WOOD   = "#55A868"   # green
C_BRICK  = "#C44E52"   # red
C_GLASS  = "#8172B3"   # purple
C_GRAY   = "#AAAAAA"

MAT_COLOR = {
    "Concrete": C_RES,
    "Steel":    C_NR,
    "Wood":     C_WOOD,
    "Brick":    C_BRICK,
    "Glass":    C_GLASS,
}

STRUCT_COLOR = {"C": C_RES, "S": C_NR, "W": C_WOOD, "B": C_BRICK, "T": C_GLASS}
STRUCT_LABEL = {"C": "Concrete-C", "S": "Steel-S", "W": "Wood-W", "B": "Brick-B", "T": "Traditional-T"}

SYSTEM_LABEL = {
    "C": "Concrete", "BC": "Brick & Concrete", "BW": "Brick & Wood",
    "W": "Wood", "SC": "Steel & Concrete", "B": "Brick",
    "S": "Steel", "T": "Traditional materials", "CW": "Concrete & Wood", "SW": "Steel & Wood",
}
SYSTEM_COLOR = {
    "C": C_RES, "BC": C_NR, "BW": "#937860", "W": C_WOOD,
    "SC": "#64B5CD", "B": C_BRICK, "S": C_GLASS, "T": C_GRAY,
    "CW": "#CCB974", "SW": "#C27D38",
}

plt.rcParams.update({
    "figure.dpi":        200,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         10,
})
random.seed(SEED); np.random.seed(SEED)


# ── data preparation (identical pipeline to prediction_model.py) ───────────
def to_period_bucket(year_series):
    year = pd.to_numeric(year_series, errors="coerce")
    return pd.cut(year, bins=[-np.inf, 1945, 1970, 1990, 2010, np.inf],
                  labels=PERIOD_BUCKETS, right=False)


def prepare_data():
    from sklearn.model_selection import train_test_split

    # Load the saved preprocessor to guarantee exact feature alignment with
    # the trained model (same OHE categories, same column order).
    preproc = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")

    df = pd.read_excel(DATA_FILE)
    df["Construction period"] = pd.to_numeric(df["Construction period"], errors="coerce")
    df["Construction period bucket"] = to_period_bucket(df["Construction period"])

    # Match the notebook's pipeline exactly: drop rows missing any X_col, then
    # keep rows with at least MIN_OBSERVED_TARGETS material values.
    df = df.dropna(subset=X_cols).reset_index(drop=True)

    mask_df = df[y_cols].notna()
    df = df.loc[mask_df.sum(axis=1) >= MIN_OBSERVED_TARGETS].reset_index(drop=True)

    X      = df[X_cols].copy()
    y_df   = df[y_cols].copy()
    y_mask = y_df.notna().to_numpy(dtype=bool)

    X_tr, X_tmp, ytr_df, ytmp_df, ytr_m, ytmp_m = train_test_split(
        X, y_df, y_mask, test_size=0.30, random_state=SEED)
    X_val, X_te, yval_df, yte_df, yval_m, yte_m = train_test_split(
        X_tmp, ytmp_df, ytmp_m, test_size=0.50, random_state=SEED)

    for d in [X_tr, X_val, X_te, ytr_df, yval_df, yte_df]:
        d.reset_index(drop=True, inplace=True)

    return dict(
        X_tr_proc  = preproc.transform(X_tr),
        X_val_proc = preproc.transform(X_val),
        X_te_proc  = preproc.transform(X_te),
        y_tr_raw   = ytr_df.to_numpy(dtype=np.float64),
        y_val_raw  = yval_df.to_numpy(dtype=np.float64),
        y_te_raw   = yte_df.to_numpy(dtype=np.float64),
        y_tr_mask  = ytr_m,
        y_val_mask = yval_m,
        y_te_mask  = yte_m,
    )


def conformal_offset(y_true, p05, p95, alpha=CAL_ALPHA):
    scores = np.maximum(p05 - y_true, y_true - p95)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 0.0
    k = int(np.ceil((scores.size + 1) * (1.0 - alpha)))
    k = max(1, min(k, scores.size))
    return float(np.sort(scores)[k - 1])


def _panel_grid(fig, n_top=3, n_bot=2, figsize=(14, 8)):
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.45)
    top = [fig.add_subplot(gs[0, i*2:(i+1)*2]) for i in range(n_top)]
    # Centre 2 bottom panels
    bot_offsets = [1, 3]
    bot = [fig.add_subplot(gs[1, o:o+2]) for o in bot_offsets]
    return top + bot


# ── load model + split data ────────────────────────────────────────────────
print("Loading model ...")
model = joblib.load(ARTIFACT_DIR / "model_finalquery.joblib")

print("Preparing data (seed=42) ...")
data = prepare_data()

int_test = model.tuned_intensity_model.predict_quantiles(data["X_te_proc"])
int_val  = model.tuned_intensity_model.predict_quantiles(data["X_val_proc"])

offsets = {}
for m, mat in enumerate(y_cols):
    obs = data["y_val_mask"][:, m]
    if obs.sum() < 2:
        offsets[mat] = 0.0
    else:
        offsets[mat] = conformal_offset(
            data["y_val_raw"][obs, m],
            int_val[mat]["p05"][obs], int_val[mat]["p95"][obs])

int_cal = {mat: {
    "p05": int_test[mat]["p05"] - offsets[mat],
    "p50": int_test[mat]["p50"],
    "p95": int_test[mat]["p95"] + offsets[mat],
} for mat in y_cols}


# ══════════════════════════════════════════════════════════════════════════════
# fig_predicted_vs_actual.png
# ══════════════════════════════════════════════════════════════════════════════
def fig_predicted_vs_actual():
    fig = plt.figure(figsize=(14, 8))
    panels = _panel_grid(fig)

    for ax, mat in zip(panels, FIG_ORDER):
        m   = y_cols.index(mat)
        obs = data["y_te_mask"][:, m]
        col = MAT_COLOR[mat]

        if obs.sum() < 2:
            ax.set_title(mat, fontsize=11); continue

        y_true = data["y_te_raw"][obs, m]
        y_pred = int_test[mat]["p50"][obs]
        mae    = mean_absolute_error(y_true, y_pred)
        r2_log = r2_score(np.log1p(y_true), np.log1p(y_pred))

        use_log = (np.percentile(y_true, 95) / (np.percentile(y_true, 5) + 1e-6)) > 10

        ax.scatter(y_true, y_pred, s=18, alpha=0.55, color=col,
                   edgecolors="none", zorder=3)

        lo = min(y_true.min(), y_pred.min()) * 0.8
        hi = max(y_true.max(), y_pred.max()) * 1.2
        if use_log:
            lo = max(lo, 1e-2)
        ax.plot([lo, hi], [lo, hi], color="#444", lw=1.2, ls="--", zorder=2)

        if use_log:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

        ax.set_title(mat, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Reported MI (kg m⁻²)", fontsize=9)
        ax.set_ylabel("Retrieved p50 (kg m⁻²)", fontsize=9)
        ax.text(0.05, 0.95, f"MAE = {mae:.1f}\nR²_log = {r2_log:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    plt.savefig(FIG_DIR / "fig_predicted_vs_actual.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print("  OK fig_predicted_vs_actual.png")


# ══════════════════════════════════════════════════════════════════════════════
# fig_coverage.png
# ══════════════════════════════════════════════════════════════════════════════
def fig_coverage():
    fig = plt.figure(figsize=(14, 8))
    panels = _panel_grid(fig)

    for ax, mat in zip(panels, FIG_ORDER):
        m   = y_cols.index(mat)
        obs = data["y_te_mask"][:, m]
        col = MAT_COLOR[mat]

        if obs.sum() < 2:
            ax.set_title(mat, fontsize=11); continue

        y_true = data["y_te_raw"][obs, m]
        p05    = int_cal[mat]["p05"][obs]
        p95    = int_cal[mat]["p95"][obs]

        order  = np.argsort(y_true)
        y_s, lo_s, hi_s = y_true[order], p05[order], p95[order]
        covered  = (y_s >= lo_s) & (y_s <= hi_s)
        cov_rate = covered.mean()
        x = np.arange(len(y_s))

        ax.fill_between(x, lo_s, hi_s, color=col, alpha=0.25, label="90% PI", zorder=1)
        ax.scatter(x[ covered],  y_s[ covered], s=8, color=col,    alpha=0.7,
                   label="Covered",    zorder=3)
        ax.scatter(x[~covered], y_s[~covered], s=8, color=C_BRICK, alpha=0.9,
                   marker="+",         label="Outside PI", zorder=4)

        ax.set_title(f"{mat}  (coverage = {cov_rate:.3f})", fontsize=11)
        ax.set_xlabel("Buildings (sorted by actual MI)", fontsize=9)
        ax.set_ylabel("MI (kg m⁻²)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")

    plt.savefig(FIG_DIR / "fig_coverage.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print("  OK fig_coverage.png")


# ══════════════════════════════════════════════════════════════════════════════
# fig1_overview.png
# ══════════════════════════════════════════════════════════════════════════════
def fig1_overview():
    df = pd.read_excel(DATA_FILE)
    df["CP"] = pd.to_numeric(df["Construction period"], errors="coerce")
    df["Decade"] = (df["CP"] // 10 * 10).astype("Int64")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.35)
    ax_A, ax_B = axes[0, 0], axes[0, 1]
    ax_C, ax_D = axes[1, 0], axes[1, 1]

    # ── (A) Construction Period ────────────────────────────────────────────
    res_mask = df[FUNC_COL] == "R"
    nr_mask  = df[FUNC_COL] == "NR"
    decades  = sorted(df["Decade"].dropna().unique())

    res_cnt = df[res_mask].groupby("Decade", observed=False).size().reindex(decades, fill_value=0)
    nr_cnt  = df[nr_mask].groupby("Decade",  observed=False).size().reindex(decades, fill_value=0)

    y_pos = np.arange(len(decades))
    ax_A.barh(y_pos, res_cnt.values, color=C_RES, label="Residential",    alpha=0.85)
    ax_A.barh(y_pos, nr_cnt.values,  color=C_NR,  label="Non-residential",
              left=res_cnt.values, alpha=0.85)
    ax_A.set_yticks(y_pos)
    ax_A.set_yticklabels([str(int(d)) for d in decades], fontsize=8)
    ax_A.set_xlabel("Number of Records", fontsize=10)
    ax_A.set_ylabel("Construction Decade", fontsize=10)
    ax_A.set_title("(A) Construction Period", fontsize=12, fontweight="bold")
    ax_A.legend(fontsize=9)

    # ── (B) Building Typology ──────────────────────────────────────────────
    typ_order = ["R-SFH", "R-MFH", "R-AB", "R-UNK",
                 "NR-OH", "NR-OL", "NR-C", "NR-E", "NR-I", "NR-P", "NR-H", "NR-UNK"]
    typ_cnt   = df["Typology"].value_counts()
    typ_vals  = [int(typ_cnt.get(t, 0)) for t in typ_order]
    typ_clrs  = [C_RES if t.startswith("R-") else C_NR for t in typ_order]
    y_b = np.arange(len(typ_order))

    ax_B.barh(y_b, typ_vals, color=typ_clrs, alpha=0.85)
    ax_B.set_yticks(y_b)
    ax_B.set_yticklabels(typ_order, fontsize=9)
    ax_B.set_xlabel("Number of Records", fontsize=10)
    ax_B.set_title("(B) Building Typology", fontsize=12, fontweight="bold")
    for y, v in zip(y_b, typ_vals):
        if v > 0:
            ax_B.text(v + max(typ_vals) * 0.01, y, str(v), va="center", fontsize=8)
    ax_B.legend(handles=[
        mpatches.Patch(color=C_RES, label="Residential"),
        mpatches.Patch(color=C_NR,  label="Non-residential"),
    ], fontsize=9)

    # ── (C) Structural System ──────────────────────────────────────────────
    sys_cnt = df["Structural_type"].value_counts()
    sys_items = [(k, int(sys_cnt[k])) for k in sys_cnt.index if k in SYSTEM_LABEL]
    sys_items = sorted(sys_items, key=lambda x: x[1], reverse=True)
    s_keys = [x[0] for x in sys_items]
    s_vals = [x[1] for x in sys_items]
    s_lbls = [SYSTEM_LABEL.get(k, k) for k in s_keys]
    s_clrs = [SYSTEM_COLOR.get(k, C_GRAY) for k in s_keys]
    y_c = np.arange(len(s_keys))

    ax_C.barh(y_c, s_vals, color=s_clrs, alpha=0.85)
    ax_C.set_yticks(y_c)
    ax_C.set_yticklabels(s_lbls, fontsize=9)
    ax_C.set_xlabel("Number of Records", fontsize=10)
    ax_C.set_title("(C) Structural System", fontsize=12, fontweight="bold")
    for y, v in zip(y_c, s_vals):
        if v > 0:
            ax_C.text(v + max(s_vals) * 0.01, y, str(v), va="center", fontsize=8)

    # ── (D) Top 20 Countries ───────────────────────────────────────────────
    ctry_cnt = df["Country"].value_counts().head(20)
    y_d = np.arange(len(ctry_cnt))
    ax_D.barh(y_d, ctry_cnt.values, color=C_RES, alpha=0.85)
    ax_D.set_yticks(y_d)
    ax_D.set_yticklabels(ctry_cnt.index, fontsize=8)
    ax_D.set_xlabel("Number of Records", fontsize=10)
    ax_D.set_title("(D) Top 20 Countries", fontsize=12, fontweight="bold")
    for y, v in zip(y_d, ctry_cnt.values):
        ax_D.text(v + max(ctry_cnt.values) * 0.01, y, str(v), va="center", fontsize=8)

    plt.savefig(FIG_DIR / "fig1_overview.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  OK fig1_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# fig2_mi_temporal.png
# ══════════════════════════════════════════════════════════════════════════════
def fig2_mi_temporal():
    df = pd.read_excel(DATA_FILE)
    df["CP"] = pd.to_numeric(df["Construction period"], errors="coerce")
    df["Decade"] = (df["CP"] // 10 * 10)
    df["Function"] = df[FUNC_COL].map({"R": "Residential", "NR": "Non-residential"})

    fig = plt.figure(figsize=(14, 8))
    panels = _panel_grid(fig)

    for ax, mat in zip(panels, FIG_ORDER):
        for func, color in [("Residential", C_RES), ("Non-residential", C_NR)]:
            sub = df[df["Function"] == func][["Decade", mat]].dropna()
            if sub.empty:
                continue
            ax.scatter(sub["Decade"], sub[mat], color=color, s=5,
                       alpha=0.12, zorder=1, edgecolors="none")
            grp    = sub.groupby("Decade")[mat]
            cnt    = grp.count()
            med    = grp.median()[cnt >= 3]
            q25    = grp.quantile(0.25)[cnt >= 3]
            q75    = grp.quantile(0.75)[cnt >= 3]
            xd     = med.index.astype(float)
            if len(xd) >= 2:
                ax.plot(xd, med.values, color=color, lw=1.8,
                        marker="o", ms=4, zorder=3, label=func)
                ax.fill_between(xd, q25.values, q75.values,
                                color=color, alpha=0.18, zorder=2)

        vals = df[mat].dropna()
        if not vals.empty:
            ax.set_ylim(bottom=0, top=vals.quantile(0.95) * 1.1)
        ax.set_xlim(left=1900)

        ax.set_title(mat, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Construction Decade", fontsize=9)
        ax.set_ylabel("MI (kg m⁻²)", fontsize=9)

    panels[0].legend(fontsize=8)
    plt.savefig(FIG_DIR / "fig2_mi_temporal.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  OK fig2_mi_temporal.png")


# ══════════════════════════════════════════════════════════════════════════════
# fig_material_covariation.png
# ══════════════════════════════════════════════════════════════════════════════
def fig_material_covariation():
    df = pd.read_excel(DATA_FILE)
    mat_order = ["Concrete", "Steel", "Wood", "Brick", "Glass"]
    n = len(mat_order)

    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    prim_codes = [c for c in ["C", "S", "W", "B", "T"]
                  if c in df["Primary Code"].dropna().values]

    for row, mat_y in enumerate(mat_order):
        for col, mat_x in enumerate(mat_order):
            ax = axes[row, col]
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False)

            if row == n - 1:
                ax.set_xlabel(f"log$_{{10}}$(1 + {mat_x})\n(kg m$^{{-2}}$)", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"log$_{{10}}$(1 + {mat_y})\n(kg m$^{{-2}}$)", fontsize=8)

            if row == col:
                # Diagonal: filled KDE per structural type
                ax.set_title(mat_x, fontsize=9, fontweight="bold")
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                for pc in prim_codes:
                    vals = df[df["Primary Code"] == pc][mat_x].dropna()
                    lv   = np.log10(1 + vals)
                    if len(lv) >= 5:
                        try:
                            kde = gaussian_kde(lv.values)
                            xs  = np.linspace(lv.min(), lv.max(), 200)
                            col_pc = STRUCT_COLOR.get(pc, C_GRAY)
                            ax.plot(xs, kde(xs), color=col_pc, lw=1.5, alpha=0.9)
                            ax.fill_between(xs, 0, kde(xs), color=col_pc, alpha=0.25)
                        except Exception:
                            pass

            elif row > col:
                # Lower triangle: scatter by structural type
                both = df[[mat_x, mat_y, "Primary Code"]].dropna()
                for pc in prim_codes:
                    sub = both[both["Primary Code"] == pc]
                    if len(sub) > 0:
                        ax.scatter(np.log10(1 + sub[mat_x]),
                                   np.log10(1 + sub[mat_y]),
                                   s=5, alpha=0.35,
                                   color=STRUCT_COLOR.get(pc, C_GRAY),
                                   edgecolors="none", zorder=2)

            else:
                # Upper triangle: Pearson r on log-transformed values
                both = df[[mat_x, mat_y]].dropna()
                nb   = len(both)
                ax.set_facecolor("#f4f4f4")
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                if nb >= 3:
                    r, p = pearsonr(np.log10(1 + both[mat_x]),
                                    np.log10(1 + both[mat_y]))
                    stars = ("***" if p < 0.001 else "**" if p < 0.01
                             else "*"   if p < 0.05  else "")
                    clr   = (C_BRICK if r > 0.25 else
                             C_RES   if r < -0.2 else "#333333")
                    ax.text(0.5, 0.60, f"r = {r:+.2f}{stars}",
                            transform=ax.transAxes, ha="center", fontsize=9,
                            fontweight="bold", color=clr)
                    ax.text(0.5, 0.35, f"n = {nb}",
                            transform=ax.transAxes, ha="center",
                            fontsize=8, color="#666")
                else:
                    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                            ha="center", fontsize=8, color=C_GRAY)

    # Legend
    handles = [mpatches.Patch(color=STRUCT_COLOR[pc], label=STRUCT_LABEL[pc])
               for pc in prim_codes]
    fig.legend(handles=handles, title="Primary Structural\nSystem Type",
               fontsize=8, title_fontsize=8,
               loc="lower right", bbox_to_anchor=(0.98, 0.02))

    plt.savefig(FIG_DIR / "fig_material_covariation.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print("  OK fig_material_covariation.png")


# ── run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nGenerating paper figures ...\n")
    fig_predicted_vs_actual()
    fig_coverage()
    fig1_overview()
    fig2_mi_temporal()
    fig_material_covariation()
    print(f"\nAll figures saved to:\n  {FIG_DIR}\n")

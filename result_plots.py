"""Presentation-quality result graphs, generated from the run's own artefacts.

These are the four figures used in the mid-semester slides. They read the CSVs
that phase1.py and phase2.py already write, so the numbers can never drift away
from the numbers in the report: re-run the phase, get the current graphs.

    fig1_phase1_ablation.png    E1-E4 forecasting error   (from phase1_summary.csv)
    fig2_phase2_forecasting.png MAE / RMSE / MAPE by model (from model_comparison.csv)
    fig3_decision_quality.png   accuracy + macro-F1 vs the majority baseline
                                                          (from classification_metrics.csv)
    fig4_error_structure.png    confusion matrices + feature-group attribution
                                (from confusion_matrix.csv, feature_importance.csv)

Called automatically at the end of a phase. Also runnable on an existing
results directory:

    python result_plots.py                # uses config.RESULTS_DIR
    python result_plots.py results_ext    # any previous run
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")                      # headless: never opens a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger("plots")

TEAL, SLATE, NAVY, AMBER = "#0E7C86", "#4A5568", "#21295C", "#D98A2B"
GREY, PALE = "#9AA5B1", "#C9CFD6"
MODEL_COLOURS = {"BiLSTM": TEAL, "GRU": SLATE, "Transformer": NAVY,
                 "Ensemble": AMBER}
CLASSES = ["SELL", "HOLD", "BUY"]
METRICS = [("MAE", "price units"), ("RMSE", "price units"), ("MAPE", "percent")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.edgecolor": SLATE, "axes.labelcolor": NAVY, "text.color": NAVY,
    "xtick.color": SLATE, "ytick.color": SLATE, "figure.autolayout": False,
})


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  plot: %s", path)
    return path


def _read(path: str) -> Optional[pd.DataFrame]:
    """Missing artefacts skip their plot instead of aborting the run."""
    if not os.path.exists(path):
        logger.warning("  plot skipped, no %s", os.path.basename(path))
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception as exc:                        # never break a finished run
        logger.warning("  plot skipped, unreadable %s (%s)",
                       os.path.basename(path), exc)
        return None


def _bar_labels(ax, bars, values, fmt="{:.4f}", rotation=0, size=9):
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center",
                va="bottom", fontsize=size, rotation=rotation, color=NAVY)


# ------------------------------------------------------------------ phase 1
def phase1_ablation(phase1_dir: str) -> Optional[str]:
    """E1-E4 forecasting error, averaged over the tickers actually run."""
    df = _read(os.path.join(phase1_dir, "phase1_summary.csv"))
    if df is None or "Experiment" not in df:
        return None
    agg = df.groupby("Experiment", sort=False)[["MAE", "RMSE", "MAPE"]].mean()
    if agg.empty:
        return None
    labels = list(agg.index)
    n_stocks = df["Stock"].nunique() if "Stock" in df else 0

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    for ax, (metric, unit) in zip(axes, METRICS):
        vals = agg[metric].to_numpy(dtype=float)
        best = int(np.argmin(vals))
        colours = [NAVY] * len(vals)
        colours[best] = AMBER
        bars = ax.bar(range(len(vals)), vals, color=colours, width=0.62,
                      edgecolor="white", linewidth=1.2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace(" + ", "\n+ ") for l in labels], fontsize=10)
        ax.set_title(f"{metric} (lower is better)", fontsize=14,
                     fontweight="bold", pad=12)
        ax.set_ylabel(unit, fontsize=11)
        lo, hi = float(vals.min()), float(vals.max())
        pad = (hi - lo) * 0.9 + 1e-9
        ax.set_ylim(lo - pad, hi + pad)
        _bar_labels(ax, bars, vals, size=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle(f"Phase 1  |  single BiLSTM, mean over {n_stocks} stock(s)  |  "
                 f"amber = best configuration", fontsize=13.5, color=NAVY, y=1.02)
    fig.tight_layout()
    return _save(fig, phase1_dir, "fig1_phase1_ablation.png")


# ------------------------------------------------------------------ phase 2
def phase2_forecasting(phase2_dir: str) -> Optional[str]:
    df = _read(os.path.join(phase2_dir, "model_comparison.csv"))
    if df is None or "Model" not in df:
        return None
    stocks = list(dict.fromkeys(df["Stock"]))
    models = list(dict.fromkeys(df["Model"]))
    xs, w = np.arange(len(stocks)), 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, (metric, unit) in zip(axes, METRICS):
        top = 0.0
        for i, m in enumerate(models):
            sub = df[df["Model"] == m].set_index("Stock")
            vals = [float(sub.loc[s, metric]) for s in stocks]
            top = max(top, max(vals))
            offset = (i - (len(models) - 1) / 2) * w
            bars = ax.bar(xs + offset, vals, w * 0.9, label=m,
                          color=MODEL_COLOURS.get(m, SLATE),
                          edgecolor="white", linewidth=1.0)
            _bar_labels(ax, bars, vals, fmt="{:.3f}", rotation=90, size=8.5)
        ax.set_xticks(xs); ax.set_xticklabels(stocks, fontsize=12)
        ax.set_title(f"{metric} (lower is better)", fontsize=14,
                     fontweight="bold", pad=10)
        ax.set_ylabel(unit, fontsize=11)
        ax.set_ylim(0, top * 1.28)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    axes[1].legend(fontsize=12, ncol=len(models), frameon=False,
                   loc="upper center", bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    return _save(fig, phase2_dir, "fig2_phase2_forecasting.png")


def decision_quality(phase2_dir: str) -> Optional[str]:
    """Ensemble accuracy and macro-F1 next to the majority-class baseline."""
    df = _read(os.path.join(phase2_dir, "classification_metrics.csv"))
    if df is None or "Macro_F1" not in df:
        return None
    ens = df[df["Model"] == "Ensemble"]
    if ens.empty:
        ens = df
    stocks = list(ens["Stock"])
    xs = np.arange(len(stocks))

    series = [("Ensemble accuracy", "Accuracy", TEAL, -0.30),
              ("Majority-class accuracy", "Majority_Acc", GREY, -0.10),
              ("Ensemble macro-F1", "Macro_F1", AMBER, 0.14),
              ("Majority-class macro-F1", "Majority_MacroF1", PALE, 0.34)]
    fig, ax = plt.subplots(figsize=(16, 5.0))
    top = 0.0
    for label, col, colour, off in series:
        if col not in ens:
            continue
        vals = ens[col].to_numpy(dtype=float)
        top = max(top, float(vals.max()))
        ax.bar(xs + off, vals, 0.19, label=label, color=colour,
               edgecolor="white", linewidth=1.0)
        bold = col in ("Macro_F1",)
        for x, v in zip(xs + off, vals):
            ax.text(x, v + 0.008, f"{v:.4f}", ha="center", fontsize=10,
                    fontweight="bold" if bold else "normal",
                    color=NAVY if bold else SLATE)
    if "Directional_Accuracy" in ens:
        for x, v in zip(xs, ens["Directional_Accuracy"].to_numpy(dtype=float)):
            ax.text(x, top * 1.16, f"Directional accuracy  {v:.4f}", ha="center",
                    fontsize=12, color=TEAL, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(stocks, fontsize=13)
    ax.set_ylim(0, top * 1.32)
    ax.set_title("Decision quality against the majority-class baseline",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11, ncol=4, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.06))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, phase2_dir, "fig3_decision_quality.png")


def error_structure(phase2_dir: str) -> Optional[str]:
    """Ensemble confusion matrices plus signed feature-group importance."""
    cm_df = _read(os.path.join(phase2_dir, "confusion_matrix.csv"))
    if cm_df is None:
        return None
    cm_df = cm_df[cm_df["Model"] == "Ensemble"] if "Model" in cm_df else cm_df
    if cm_df.empty:
        return None
    stocks = list(cm_df["Stock"])
    imp_df = _read(os.path.join(phase2_dir, "feature_importance.csv"))
    show_imp = imp_df is not None and "Group" in imp_df

    n = len(stocks)
    widths = [1.0] * n + ([1.5] if show_imp else [])
    fig, axes = plt.subplots(1, len(widths), figsize=(4.7 * len(widths), 5.0),
                             gridspec_kw={"width_ratios": widths, "wspace": 0.62})
    axes = np.atleast_1d(axes)

    for j, stock in enumerate(stocks):
        row = cm_df[cm_df["Stock"] == stock].iloc[0]
        m = np.array([[float(row.get(f"true_{t}_pred_{p}", 0))
                       for p in CLASSES] for t in CLASSES])
        a = axes[j]
        denom = m.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        a.imshow(m / denom, cmap="BuPu", vmin=0, vmax=0.75)
        for r in range(3):
            for c in range(3):
                a.text(c, r, f"{int(m[r, c])}", ha="center", va="center",
                       fontsize=15, fontweight="bold",
                       color="white" if m[r, c] / denom[r, 0] > 0.45 else NAVY)
        a.set_xticks(range(3)); a.set_xticklabels(CLASSES, fontsize=11.5)
        a.set_yticks(range(3)); a.set_yticklabels(CLASSES, fontsize=11.5)
        a.set_xlabel("predicted", fontsize=11.5)
        if j == 0:
            a.set_ylabel("true", fontsize=11.5)
        a.set_title(f"{stock}  (ensemble)", fontsize=13, fontweight="bold", pad=10)
        for sp in a.spines.values():
            sp.set_visible(False)

    if show_imp:
        a2 = axes[-1]
        groups = list(dict.fromkeys(imp_df["Group"]))
        ys, h = np.arange(len(groups)), 0.8 / max(len(stocks), 1)
        for i, stock in enumerate(stocks):
            sub = imp_df[imp_df["Stock"] == stock].set_index("Group")
            vals = [float(sub.loc[g, "Importance_Pct"]) if g in sub.index else 0.0
                    for g in groups]
            a2.barh(ys + (i - (len(stocks) - 1) / 2) * h, vals, h * 0.9,
                    label=stock, color=[TEAL, NAVY, AMBER, SLATE][i % 4],
                    edgecolor="white", linewidth=1.0)
        a2.axvline(0, color=SLATE, lw=1.2)
        a2.set_yticks(ys)
        a2.set_yticklabels([g.replace(" ", "\n", 1) for g in groups], fontsize=11)
        a2.invert_yaxis()
        a2.set_xlabel("signed permutation importance on validation (%)", fontsize=11)
        a2.set_title("Feature-group attribution", fontsize=13, fontweight="bold",
                     pad=10)
        a2.legend(fontsize=11, ncol=len(stocks), frameon=False,
                  loc="upper center", bbox_to_anchor=(0.5, -0.13))
        a2.spines[["top", "right", "left"]].set_visible(False)
        a2.grid(axis="x", alpha=0.25); a2.set_axisbelow(True)
    # imshow axes are not tight_layout friendly; the gridspec spacing above
    # already does the job, so only the outer margins are trimmed
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.16)
    return _save(fig, phase2_dir, "fig4_error_structure.png")


# ------------------------------------------------------------------ entry points
def save_phase1_plots(phase1_dir: str) -> None:
    logger.info("generating Phase-1 presentation graphs")
    try:
        phase1_ablation(phase1_dir)
    except Exception as exc:                 # a plot must never fail a finished run
        logger.warning("Phase-1 plots failed (%s: %s)", type(exc).__name__, exc)


def save_phase2_plots(phase2_dir: str) -> None:
    logger.info("generating Phase-2 presentation graphs")
    for fn in (phase2_forecasting, decision_quality, error_structure):
        try:
            fn(phase2_dir)
        except Exception as exc:
            logger.warning("%s failed (%s: %s)", fn.__name__,
                           type(exc).__name__, exc)


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else None
    if root is None:
        from config import RESULTS_DIR
        root = RESULTS_DIR
    p1, p2 = os.path.join(root, "phase1"), os.path.join(root, "phase2")
    if os.path.isdir(p1):
        save_phase1_plots(p1)
    if os.path.isdir(p2):
        save_phase2_plots(p2)
    if not (os.path.isdir(p1) or os.path.isdir(p2)):
        logger.error("no phase1/ or phase2/ directory under %s", root)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

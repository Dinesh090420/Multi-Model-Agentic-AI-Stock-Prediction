"""Reduced output layer: ONLY the 5 forecasting results.

Everything else that used to be produced here (classification/direction bars,
sentiment over time, sentiment distribution, walk-forward, return-direction
scatter, market regime, social engagement, prediction error, cross-sectional
ranking, dashboard, data-collection summary, sentiment summary) has been
removed. Metrics are restricted to MAE / RMSE / MAPE.

The 5 kept results
------------------
 1  Actual vs Predicted            plots/{T}_01_actual_vs_predicted.png (+ .csv)
 2  Forecast error by model        plots/{T}_02_forecast_error.png
 3  Training/validation summary    plots/{T}_03_loss_curves.png
 4  Metrics table (MAE/RMSE/MAPE)  plots/{T}_04_metrics_table.png
                                   metrics/{T}_metrics.csv
 5  Final overall result           plots/00_final_forecasting_summary.png
                                   metrics/final_forecasting_summary.csv

Console output is printed by print_final_summary() at the end of the run.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import rpath, get_logger
from agents import MODEL_NAMES

logger = get_logger("visualize")

# The persistence Baseline is kept because MAE/RMSE are in dollars and are
# uninterpretable without a reference. Set to False for a strictly 4-row table.
INCLUDE_BASELINE = True

METRIC_KEYS = ["MAE", "RMSE", "MAPE"]          # the only metrics reported now
MODELS = (["Baseline"] if INCLUDE_BASELINE else []) + MODEL_NAMES + ["Ensemble"]
MODEL_COLORS = {"Actual": "#222222", "LSTM": "#1f77b4", "GRU": "#2ca02c",
                "Transformer": "#9467bd", "Ensemble": "#d62728",
                "Baseline": "#7f7f7f"}


def _save(fig, sub, name):
    fig.tight_layout()
    fig.savefig(rpath(sub, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _table_fig(df, title, figsize=None):
    rows, cols = df.shape
    fig, ax = plt.subplots(figsize=figsize or (min(2 + 2.0 * cols, 18),
                                               1.2 + 0.45 * rows))
    ax.axis("off")
    ax.set_title(title, fontweight="bold", pad=12)
    tbl = ax.table(cellText=df.astype(str).values, colLabels=df.columns,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(cols):
        tbl[0, j].set_facecolor("#40466e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    return fig


def _metrics_frame(b):
    """Model x {MAE, RMSE, MAPE} for one ticker - values copied, never recomputed."""
    return pd.DataFrame({m: {k: round(float(b["metrics"][m][k]), 4)
                             for k in METRIC_KEYS} for m in MODELS}).T


# ============================================================ 1. actual vs predicted
def plot_actual_vs_predicted(b):
    t, x = b["ticker"], b["test_dates"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, b["actual_close"], label="Actual", color=MODEL_COLORS["Actual"], lw=2)
    for m in MODEL_NAMES + ["Ensemble"]:
        ax.plot(x, b["preds"][m], label=m, color=MODEL_COLORS[m], lw=1.2, alpha=0.85)
    ax.set(title=f"{t} - Actual vs Predicted Close (test set)",
           xlabel="Date", ylabel="Price ($)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "plots", f"{t}_01_actual_vs_predicted.png")
    pd.DataFrame({"Date": x, "Actual": b["actual_close"],
                  **{m: b["preds"][m] for m in MODEL_NAMES + ["Ensemble"]}}
                 ).to_csv(rpath("plots", f"{t}_01_actual_vs_predicted.csv"),
                          index=False)


# ============================================================ 2. forecast error
def plot_forecast_error(b):
    """MAE / RMSE / MAPE side by side. MAPE is on a twin axis (percent, not $)."""
    t = b["ticker"]
    df = _metrics_frame(b)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    units = {"MAE": "$", "RMSE": "$", "MAPE": "%"}
    for ax, k in zip(axes, METRIC_KEYS):
        vals = df[k].to_numpy(dtype=float)
        bars = ax.bar(df.index, vals, color=[MODEL_COLORS[m] for m in df.index])
        best = int(np.argmin(vals))
        bars[best].set_edgecolor("black"); bars[best].set_linewidth(2.0)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set(title=f"{k} ({units[k]}) - lower is better", ylabel=k)
        ax.tick_params(axis="x", rotation=30)
        ax.margins(y=0.18)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{t} - Forecast Error by Model (test set)", fontweight="bold")
    _save(fig, "plots", f"{t}_02_forecast_error.png")


# ============================================================ 3. train/val summary
def plot_loss_curves(b):
    """Training vs validation MSE per epoch + the chronological split sizes."""
    t = b["ticker"]
    d = b.get("data", {})
    n_tr, n_va, N = d.get("n_train"), d.get("n_val"), d.get("N")
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(5 * len(MODEL_NAMES), 4.3))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, MODEL_NAMES):
        h = b["histories"][m]
        ax.plot(h["train"], label="train")
        if h.get("val"):
            ax.plot(h["val"], label="validation")
        ax.set(title=m, xlabel="Epoch", ylabel="MSE loss (scaled return)")
        ax.legend(); ax.grid(alpha=0.3)
    sub = (f"train {n_tr} / val {n_va} / test {N - n_tr - n_va} sequences "
           f"(chronological)" if None not in (n_tr, n_va, N) else "")
    fig.suptitle(f"{t} - Training vs Validation Loss   {sub}", fontweight="bold")
    _save(fig, "plots", f"{t}_03_loss_curves.png")


# ============================================================ 4. metrics table
def save_metrics_table(b):
    t = b["ticker"]
    df = _metrics_frame(b).reset_index().rename(columns={"index": "Model"})
    df.to_csv(rpath("metrics", f"{t}_metrics.csv"), index=False)
    fig = _table_fig(df, f"{t} - Forecasting Metrics (test set)")
    _save(fig, "plots", f"{t}_04_metrics_table.png")
    return df


def per_ticker_plots(b):
    plot_actual_vs_predicted(b)
    plot_forecast_error(b)
    plot_loss_curves(b)
    save_metrics_table(b)


# ============================================================ 5. final results
def _summary_frame(results):
    """All datasets x all models, MAE/RMSE/MAPE. Values copied, never recomputed."""
    rows = []
    for b in results:
        for m in MODELS:
            row = {"Dataset": b["ticker"], "Model": m}
            row.update({k: round(float(b["metrics"][m][k]), 4) for k in METRIC_KEYS})
            rows.append(row)
    return pd.DataFrame(rows)


def _best_per_dataset(df):
    """Lowest-MAPE model per dataset. Selection only - no value is recomputed."""
    return df.loc[df.groupby("Dataset")["MAPE"].idxmin()].set_index("Dataset")


def final_forecasting_summary(results):
    """One CSV + one rendered table holding every final metric value."""
    df = _summary_frame(results)
    df.to_csv(rpath("metrics", "final_forecasting_summary.csv"), index=False)
    fig = _table_fig(df, "FINAL FORECASTING METRICS - all datasets, all models",
                     figsize=(12, 1.5 + 0.42 * len(df)))
    _save(fig, "plots", "00_final_forecasting_summary.png")
    return df, _best_per_dataset(df)


# ------------------------------------------------------------------ console
_W = 66


def _rule(ch="-"):
    return ch * _W


def _metric_block(sub, indent="  ", star_model=None):
    out = [f"{indent}{'Model':<13}{'MAE':>11}{'RMSE':>11}{'MAPE %':>11}",
           indent + _rule("-")[:len(indent) + 46 - len(indent)]]
    for _, r in sub.iterrows():
        mark = " *" if r["Model"] == star_model else ""
        out.append(f"{indent}{r['Model']:<13}{r['MAE']:>11.4f}"
                   f"{r['RMSE']:>11.4f}{r['MAPE']:>11.4f}{mark}")
    return out


def _dataset_section(b, sub, best_model):
    """One self-contained block per dataset: what it was trained on, then models."""
    t = b["ticker"]
    si = b.get("src_info", {}) or {}
    d = b.get("data", {}) or {}
    n_tr, n_va, N = d.get("n_train"), d.get("n_val"), d.get("N")
    td = b.get("test_dates")

    L = ["", _rule("="), f"DATASET: {t}", _rule("=")]
    if si.get("source"):
        L.append(f"  Source        : {si['source']}")
    if si.get("records"):
        L.append(f"  OHLCV records : {si['records']}   range {si.get('date_range', '-')}")
    if None not in (n_tr, n_va, N):
        L.append(f"  Split         : train {n_tr} / val {n_va} / test {N - n_tr - n_va}"
                 f"  (chronological, no shuffling)")
    if td is not None and len(td):
        L.append(f"  Test period   : {pd.Timestamp(td[0]).date()} -> "
                 f"{pd.Timestamp(td[-1]).date()}  ({len(td)} days)")
    L += ["", "  MODELS (test-set forecasting error)"]
    L += _metric_block(sub, indent="    ", star_model=best_model)
    L.append(f"    Best on this dataset: {best_model}")
    return L


def print_final_summary(results):
    """Per-dataset sections first, then one combined final-metrics table."""
    df, best = final_forecasting_summary(results)

    lines = []
    for b in results:
        t = b["ticker"]
        sub = df[df["Dataset"] == t]
        lines += _dataset_section(b, sub, best.loc[t, "Model"])

    lines += ["", _rule("="), "FINAL RESULTS - ALL DATASETS".center(_W), _rule("="),
              f"  {'Dataset':<10}{'Model':<13}{'MAE':>11}{'RMSE':>11}{'MAPE %':>11}",
              "  " + _rule("-")[:56]]
    for tick, sub in df.groupby("Dataset", sort=False):
        for _, r in sub.iterrows():
            mark = " *" if r["Model"] == best.loc[tick, "Model"] else ""
            lines.append(f"  {r['Dataset']:<10}{r['Model']:<13}{r['MAE']:>11.4f}"
                         f"{r['RMSE']:>11.4f}{r['MAPE']:>11.4f}{mark}")
        lines.append("")
    wins = best["Model"].value_counts()
    lines += [_rule("-"),
              "  Best per dataset (*): " +
              ", ".join(f"{t}={best.loc[t, 'Model']}" for t in best.index),
              f"  OVERALL BEST MODEL  : {wins.index[0]} "
              f"(lowest MAPE on {int(wins.iloc[0])}/{len(best)} datasets)",
              _rule("="), ""]
    logger.info("\n".join(lines))
    return df


# ==================================================================== 4-section
# Component-separated report: YFinance / NewsAPI / StockTwits / Hybrid.
# Nothing here is recomputed - every number is copied from the run, and any
# result the project does not produce is printed as NOT AVAILABLE.
NA = "Not available in the current results."
PRICE_ONLY_CSV = "price_only_summary.csv"   # written by a PRICE_ONLY ablation run


def _section(title, num):
    return ["", "=" * _W, f"{num}. {title}", "=" * _W]


def _sec1_yfinance():
    """Price-only forecasting. Exists only if a PRICE_ONLY ablation was run."""
    L = _section("YFINANCE - STOCK FORECASTING (price features only)", 1)
    path = rpath("metrics", PRICE_ONLY_CSV)
    if not os.path.exists(path):
        L += ["  " + NA,
              "  Reason: every trained model in this run used the FUSED feature",
              "          matrix (price + technical + Elliott Wave + sentiment +",
              "          regime). No price-only ablation has been run, so no",
              "          YFinance-only MAE/RMSE/MAPE exists.",
              f"  To populate: run the price-only ablation, save its per-model",
              f"          metrics to results/metrics/{PRICE_ONLY_CSV}"]
        return L
    df = pd.read_csv(path)
    L += _metric_block(df, indent="  ")
    for k in METRIC_KEYS:
        r = df.loc[df[k].idxmin()]
        L.append(f"  Best {k:<6}: {r[k]:.4f}  ({r['Model']}"
                 + (f", {r['Dataset']}" if "Dataset" in df.columns else "") + ")")
    return L


def _sec_text(num, title, info, scored, key, coverage, live_info):
    """Shared body for the NewsAPI (2) and StockTwits (3) sections."""
    L = _section(title, num)
    L += ["  EVALUATION METRIC",
          "    " + NA,
          "    FinBERT is applied without labelled ground truth, so no accuracy,",
          "    precision, recall or F1 can be computed for this component.",
          "", "  AVAILABLE DESCRIPTIVE RESULTS (not evaluation metrics)"]
    L.append(f"    Training source   : {info.get('source', '-')}")
    L.append(f"    Items scored      : {info.get(key, 0)}")
    if live_info is not None:
        L.append(f"    Live source       : {live_info.get('source', '-')}  "
                 f"({live_info.get(key, 0)} items)")
    if scored is not None and len(scored) and "FinBERT_Label" in scored:
        vc = scored["FinBERT_Label"].value_counts()
        L.append("    FinBERT classes   : " +
                 ", ".join(f"{k}={int(v)}" for k, v in vc.items()))
    if coverage:
        L.append("    Split coverage    : " +
                 ", ".join(f"{k} {v*100:.0f}%" for k, v in coverage.items()))
    return L


def _sec4_hybrid(df, best):
    L = _section("FINAL HYBRID SYSTEM - COMBINED RESULTS", 4)
    L += ["  price + technical indicators + Elliott Wave + sentiment + regime",
          "  -> feature fusion -> LSTM + GRU + Transformer -> Ensemble", "",
          f"  {'Dataset':<10}{'Model':<13}{'MAE':>11}{'RMSE':>11}{'MAPE %':>11}",
          "  " + _rule("-")[:56]]
    for tick, sub in df.groupby("Dataset", sort=False):
        for _, r in sub.iterrows():
            mark = " *" if r["Model"] == best.loc[tick, "Model"] else ""
            L.append(f"  {r['Dataset']:<10}{r['Model']:<13}{r['MAE']:>11.4f}"
                     f"{r['RMSE']:>11.4f}{r['MAPE']:>11.4f}{mark}")
        L.append("")
    wins = best["Model"].value_counts()
    L += [f"  Best per dataset (*): " +
          ", ".join(f"{t}={best.loc[t, 'Model']}" for t in best.index),
          f"  OVERALL BEST MODEL  : {wins.index[0]} "
          f"(lowest MAPE on {int(wins.iloc[0])}/{len(best)} datasets)"]
    return L


def print_report(results):
    """The full four-section, component-separated results report."""
    df, best = final_forecasting_summary(results)
    b0 = results[0]
    live = b0.get("live") or {}

    lines = ["", "#" * _W, "COMPONENT-SEPARATED RESULTS REPORT".center(_W), "#" * _W]
    lines += _sec1_yfinance()
    lines += _sec_text(2, "NEWSAPI - NEWS / SENTIMENT RESULTS",
                       b0.get("news_info", {}), b0.get("news_scored"), "articles",
                       None, live.get("news_info"))
    lines += _sec_text(3, "STOCKTWITS - SOCIAL / SENTIMENT RESULTS",
                       b0.get("social_info", {}), b0.get("social_scored"), "posts",
                       b0.get("sent_coverage"), live.get("social_info"))
    lines += _sec4_hybrid(df, best)
    lines += ["", "#" * _W, ""]

    text = "\n".join(lines)
    logger.info(text)
    with open(rpath("metrics", "component_results_report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    return df


def universe_outputs(results, rank_df=None):
    """Kept for call compatibility; rank_df is ignored (ranking output removed)."""
    return print_report(results)

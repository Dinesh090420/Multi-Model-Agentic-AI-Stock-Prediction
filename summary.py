"""One-page summary of everything, read from files already on disk.

Trains nothing and changes nothing - it just loads the CSVs written by
main.py and phase2.py and prints outlook, decision, forecasting and
classification for all stocks in one view.

    python summary.py                 # reads results_ext/ (extended study)
    RESULTS=results python summary.py # reads results/    (2021-22 study)

OUTLOOK is defined explicitly, not guessed:
    bullish/bearish = the majority BUY vs SELL among the model's predicted
    signals over the last OUTLOOK_DAYS of the test set, shown next to what
    the actual market did over the same days so the two can be compared.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.getenv("RESULTS", "results_ext")
OUTLOOK_DAYS = 20
W = 78


def _load(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def outlook(sub: pd.DataFrame, col: str) -> str:
    """Majority signal over the final OUTLOOK_DAYS. Explicit, reproducible."""
    tail = sub.tail(OUTLOOK_DAYS)
    b, s = int((tail[col] == "BUY").sum()), int((tail[col] == "SELL").sum())
    if b > s:
        return f"BULLISH ({b} BUY vs {s} SELL)"
    if s > b:
        return f"BEARISH ({s} SELL vs {b} BUY)"
    return f"NEUTRAL ({b} BUY, {s} SELL)"


def main() -> int:
    p2, p1 = os.path.join(ROOT, "phase2"), os.path.join(ROOT, "phase1")
    preds = _load(os.path.join(p2, "ensemble_predictions.csv"))
    fore = _load(os.path.join(p2, "model_comparison.csv"))
    cls = _load(os.path.join(p2, "classification_metrics.csv"))
    ph1 = _load(os.path.join(p1, "phase1_summary.csv"))
    if preds is None or fore is None or cls is None:
        print(f"No Phase-2 results under {p2}/. Run phase2.py first.")
        return 1

    print("\n" + "#" * W)
    print(f"  FINAL SUMMARY - {ROOT}".ljust(W))
    print("#" * W)

    for stock in preds["Stock"].unique():
        sub = preds[preds["Stock"] == stock].reset_index(drop=True)
        last = sub.iloc[-1]
        ens_f = fore[(fore["Stock"] == stock) & (fore["Model"] == "Ensemble")].iloc[0]
        ens_c = cls[(cls["Stock"] == stock) & (cls["Model"] == "Ensemble")].iloc[0]
        hit = sub["Predicted_Signal"] == sub["True_Signal"]

        print("\n" + "=" * W)
        print(f"  {stock}")
        print("=" * W)
        print(f"  OUTLOOK (last {OUTLOOK_DAYS} test days)")
        print(f"    Model says   : {outlook(sub, 'Predicted_Signal')}")
        print(f"    Market did   : {outlook(sub, 'True_Signal')}")

        print(f"\n  LATEST DECISION  ({pd.Timestamp(last['Date']).date()})")
        ok = "correct" if last["Predicted_Signal"] == last["True_Signal"] else "wrong"
        print(f"    Signal       : {last['Predicted_Signal']}   "
              f"(actual {last['True_Signal']} - {ok})")
        print(f"    Return       : {last['Predicted_Return'] * 100:+.2f}%  ->  "
              f"close {last['Predicted_Close']:.2f} (actual {last['Actual_Close']:.2f})")
        print(f"    Confidence   : {last['Confidence'] * 100:.0f}%")
        votes = ", ".join(f"{m.replace('Pred_', '')} {last[m] * 100:+.2f}%"
                          for m in sub.columns if m.startswith("Pred_"))
        print(f"    Model votes  : {votes}")

        print("\n  FORECASTING (ensemble, test set)")
        print(f"    MAE {ens_f['MAE']:.4f}   RMSE {ens_f['RMSE']:.4f}   "
              f"MAPE {ens_f['MAPE']:.4f}%")
        if ph1 is not None:
            e1 = ph1[(ph1["Stock"] == stock) & (ph1["Exp"] == "E1")]
            if not e1.empty:
                r = e1.iloc[0]
                print(f"    Phase-1 baseline (YFinance only): MAE {r['MAE']:.4f}   "
                      f"RMSE {r['RMSE']:.4f}   MAPE {r['MAPE']:.4f}%")

        print("\n  CLASSIFICATION (BUY / SELL / HOLD, test set)")
        print(f"    Correct      : {int(hit.sum())} of {len(sub)} days "
              f"({hit.mean() * 100:.1f}%)   [random = 33%]")
        print(f"    Accuracy {ens_c['Accuracy']:.4f}   Macro-P "
              f"{ens_c['Macro_Precision']:.4f}   Macro-R {ens_c['Macro_Recall']:.4f}"
              f"   Macro-F1 {ens_c['Macro_F1']:.4f}")
        print(f"    Direction    : {ens_c['Directional_Accuracy']:.4f}   "
              f"[coin flip = 0.50]")
        per = "   ".join(
            f"{c} {int(((sub['True_Signal'] == c) & hit).sum())}/"
            f"{int((sub['True_Signal'] == c).sum())}" for c in ("SELL", "HOLD", "BUY"))
        print(f"    By class     : {per}")

    print("\n" + "=" * W)
    print("  ACROSS ALL STOCKS")
    print("=" * W)
    print(f"  {'Stock':<8}{'Outlook':<12}{'MAPE %':>9}{'Correct':>10}"
          f"{'Macro-F1':>10}{'Direction':>11}")
    print("  " + "-" * (W - 4))
    for stock in preds["Stock"].unique():
        sub = preds[preds["Stock"] == stock]
        hit = (sub["Predicted_Signal"] == sub["True_Signal"]).mean()
        f = fore[(fore["Stock"] == stock) & (fore["Model"] == "Ensemble")].iloc[0]
        c = cls[(cls["Stock"] == stock) & (cls["Model"] == "Ensemble")].iloc[0]
        print(f"  {stock:<8}{outlook(sub, 'Predicted_Signal').split(' ')[0]:<12}"
              f"{f['MAPE']:>9.4f}{hit * 100:>9.1f}%{c['Macro_F1']:>10.4f}"
              f"{c['Directional_Accuracy']:>11.4f}")
    print("\n" + "#" * W + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

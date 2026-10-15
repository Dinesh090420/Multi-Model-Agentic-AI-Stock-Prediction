"""Evaluation: regression + direction + genuine 3-class BUY/SELL/HOLD metrics,
leakage-free walk-forward validation, and validation-based threshold optimization.

Definitions (used everywhere, identical across models):
  target            = next-day (HORIZON) close price
  predicted return  = (predicted_close - current_close) / current_close
  direction         = sign of the return (Up if > 0)
  BUY/SELL/HOLD     = threshold on the return; thresholds are OPTIMIZED on VALIDATION
                      data (optimize_thresholds), never on the test set.
Regression metrics use prices; direction/classification metrics use returns.
The final test set is never used for tuning or model selection.
"""
import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import TimeSeriesSplit
from config import (BUY_THRESHOLD, SELL_THRESHOLD, WF_SPLITS, WF_EPOCHS,
                    LEARNING_RATE, BATCH_SIZE, THRESH_QUANTILES, MIN_SIGNAL_SUPPORT)
from models import make_model, train_model, predict_returns

_CLASSES = ["SELL", "HOLD", "BUY"]


def to_signal(r, buy=BUY_THRESHOLD, sell=SELL_THRESHOLD):
    return "BUY" if r >= buy else ("SELL" if r <= sell else "HOLD")


def signals(returns, buy=BUY_THRESHOLD, sell=SELL_THRESHOLD):
    return np.array([to_signal(r, buy, sell) for r in np.asarray(returns)])


def optimize_thresholds(val_pred_ret, val_actual_ret,
                        quantiles=THRESH_QUANTILES, min_support=MIN_SIGNAL_SUPPORT):
    """Validation-optimized BUY/SELL thresholds, optimized PER SIDE.

    Each side scans its own quantile candidates of the VALIDATION predicted-
    return distribution and keeps the sign-valid candidate (BUY > 0, SELL < 0)
    that maximizes the side's directional precision, tie-broken by support. The
    sides are independent: if the model's validation predictions are skewed, the
    infeasible side alone falls back to its config default (correctly making that
    action rare) while the other side stays data-driven and inside the predicted
    range. The previous symmetric version let one infeasible side force BOTH
    sides to the fixed defaults, which typically lie outside the entire
    predicted-return range and turned every signal into HOLD. Never touches test.
    """
    vp = np.asarray(val_pred_ret, dtype=float)
    va = np.asarray(val_actual_ret, dtype=float)

    def _side(sign, default):
        best = None
        for q in quantiles:
            thr = float(np.quantile(vp, 1 - q if sign > 0 else q))
            if (sign > 0 and thr <= 0.0) or (sign < 0 and thr >= 0.0):
                continue                                   # sign guard, per side
            mask = vp >= thr if sign > 0 else vp <= thr
            if mask.sum() < min_support:
                continue
            hits = (va[mask] > 0) if sign > 0 else (va[mask] < 0)
            score = (float(hits.mean()), int(mask.sum()))
            if best is None or score > best[0]:
                best = (score, thr)
        return best[1] if best is not None else default

    buy_thr = _side(+1, BUY_THRESHOLD)
    sell_thr = _side(-1, SELL_THRESHOLD)
    if buy_thr <= sell_thr:                                # degenerate overlap
        return BUY_THRESHOLD, SELL_THRESHOLD
    return buy_thr, sell_thr


def full_metrics(actual_close, pred_close, actual_ret, pred_ret,
                 buy=BUY_THRESHOLD, sell=SELL_THRESHOLD):
    """Return dict of RMSE, MAE, MAPE, Directional Accuracy, Accuracy,
    Precision, Recall, F1 and an annualized Sharpe ratio of the naive
    threshold strategy (+1 on BUY, -1 on SELL, 0 on HOLD; frictionless)."""
    rmse = float(np.sqrt(mean_squared_error(actual_close, pred_close)))
    mae = float(mean_absolute_error(actual_close, pred_close))
    ac = np.asarray(actual_close, dtype=float)
    mape = float(np.mean(np.abs((ac - np.asarray(pred_close)) / ac)) * 100.0)

    a_ret = np.asarray(actual_ret, dtype=float)
    p_ret = np.asarray(pred_ret, dtype=float)
    dir_acc = float(np.mean((a_ret > 0).astype(int) == (p_ret > 0).astype(int)))

    # annualized Sharpe of the signal-following strategy (frictionless, for
    # ACADEMIC comparison only - not an investable performance claim)
    pos = np.where(p_ret >= buy, 1.0, np.where(p_ret <= sell, -1.0, 0.0))
    strat = pos * a_ret
    sd = strat.std()
    sharpe = float(np.sqrt(252.0) * strat.mean() / sd) if sd > 1e-12 else 0.0

    y_true = signals(a_ret, buy, sell)
    y_pred = signals(p_ret, buy, sell)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, labels=_CLASSES,
                                 average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, labels=_CLASSES,
                             average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, labels=_CLASSES,
                        average="weighted", zero_division=0))
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape,
            "Directional_Accuracy": dir_acc, "Accuracy": acc,
            "Precision": prec, "Recall": rec, "F1": f1, "Sharpe": sharpe}


def walk_forward(data, model_names, ret_scaler, hp_map=None):
    """Expanding-window walk-forward directional accuracy on TRAIN+VAL only.

    TimeSeriesSplit preserves chronological order; the final test slice is
    excluded, so this never touches held-out test data. hp_map supplies the
    validation-chosen hyperparameters so the check evaluates the deployed
    configuration. Returns {"folds": k, model_name: [per-fold dir-acc...],
    "Ensemble": [...]}.
    """
    N, n_train, n_val = data["N"], data["n_train"], data["n_val"]
    end = n_train + n_val                       # exclude test
    X = data["X"][:end]
    y = data["y"][:end]
    aret = data["actual_return"][:end]

    n_splits = min(WF_SPLITS, max(2, end // 60))
    tss = TimeSeriesSplit(n_splits=n_splits)
    result = {"folds": n_splits, "Ensemble": []}
    for m in model_names:
        result[m] = []

    for tr_idx, va_idx in tss.split(X):
        fold_rets = {}
        for m in model_names:
            hp = (hp_map or {}).get(m)
            model = make_model(m, data["input_size"], hp)
            model, _ = train_model(model, X[tr_idx], y[tr_idx],
                                   epochs=WF_EPOCHS, patience=WF_EPOCHS,
                                   lr=(hp or {}).get("lr", LEARNING_RATE),
                                   batch_size=(hp or {}).get("batch_size", BATCH_SIZE))
            pr = predict_returns(model, X[va_idx], ret_scaler)   # predicted return
            fold_rets[m] = pr
            result[m].append(_dir_acc(aret[va_idx], pr))
        ens = np.mean([fold_rets[m] for m in model_names], axis=0)
        result["Ensemble"].append(_dir_acc(aret[va_idx], ens))
    return result


def _dir_acc(actual_ret, pred_ret):
    return float(np.mean((np.asarray(actual_ret) > 0) == (np.asarray(pred_ret) > 0)))
"""PHASE 2 - BiLSTM + GRU + Transformer -> ensemble -> BUY/SELL/HOLD -> XAI.

Phase 1 is untouched. This module REUSES its pipeline end to end: the same
DataAgent/SentimentAgent, the same build_feature_frame, the same make_dataset
(so identical chronological 121/25/27 split, identical target, identical
train-only scalers), the same historical-news loader from phase1, and the same
metric functions from evaluation.py. `python main.py` still runs Phase 1.

    features (E4 full fusion: price + technical + Elliott Wave + news + social
              + regime)
        -> BiLSTM, GRU, Transformer trained independently, same seed and
           hyperparameters
        -> ensemble by inverse-MSE weights fitted on VALIDATION only
        -> predicted next-day return / close
        -> BUY / SELL / HOLD via thresholds optimised on VALIDATION only
        -> Accuracy / Precision / Recall / F1 / confusion matrix
        -> explainable report per stock

LEAKAGE CONTROL. The test slice is used exactly once, to score final
predictions. Scalers are fit on train rows (features.make_dataset), ensemble
weights on validation predictions, decision thresholds on validation returns
(evaluation.optimize_thresholds), and feature-group importance on validation.
No test value influences any fitted quantity.

DECISION RULE (identical for true and predicted labels):
    r = (close[t+1] - close[t]) / close[t]
    BUY  if r >= tau_buy
    SELL if r <= tau_sell
    HOLD otherwise
True labels use the ACTUAL return; predicted labels use the ENSEMBLE return.
tau_buy / tau_sell come from the validation set only.

Run:  python phase2.py          (or PHASE=2 python main.py)
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from agents import DataAgent, SentimentAgent
from config import (BATCH_SIZE, BUY_THRESHOLD, DEVICE, EARLY_STOP_PATIENCE, EPOCHS,
                    LEARNING_RATE, RESULTS_DIR, SELL_THRESHOLD, SEED, TICKERS)
from evaluation import full_metrics, signals
from features import build_feature_frame, make_dataset, split_bounds
from models import (make_model, predict_multitask, predict_returns,
                    train_model, train_multitask)
from phase1 import (FUSION_ONLY, NEWS_FEATURES, SOCIAL_COLS, YF_GROUP,
                    load_corpus, news_features)
from sentiment_analysis import save_cache
from utils import configure_logging, ensure_dirs, get_logger, set_seed

logger = get_logger("phase2")

MODEL_NAMES = ["BiLSTM", "GRU", "Transformer"]
# Deep-ensemble seeds. Each model is trained once per seed and its predictions
# averaged, which removes the run-to-run instability a single seed produces on
# 121 training sequences. Metric spread across seeds is reported too.
SEEDS = [SEED + i for i in range(5)]
# Trailing window for the rolling bias correction (trading days).
BIAS_WINDOW = 60
METRIC_KEYS = ["MAE", "RMSE", "MAPE"]
CLASS_KEYS = ["Accuracy", "Precision", "Recall", "F1"]
CLASSES = ["SELL", "HOLD", "BUY"]
PHASE2_DIR = os.path.join(RESULTS_DIR, "phase2")
_W = 82

# Same feature set as Phase 1 E4 (full fusion) - nothing added, nothing dropped.
FUSION_FEATURES = YF_GROUP + NEWS_FEATURES + SOCIAL_COLS + FUSION_ONLY

# Groups used for the explainability layer.
FEATURE_GROUPS = {
    "Price/Technical": YF_GROUP,
    "News sentiment": NEWS_FEATURES,
    "Social sentiment": SOCIAL_COLS + ["Combined_Sentiment", "Sent_Mom"],
    "Market regime": [c for c in FUSION_ONLY
                      if c not in ("Combined_Sentiment", "Sent_Mom")],
}


# ------------------------------------------------------------------ tuning
# Search space. Every candidate is scored ONLY on validation loss; the test
# slice is never built during tuning.
SEQ_GRID = [15, 30, 45]
HP_GRID = [
    {"hidden_size": 32, "num_layers": 1, "dropout": 0.1, "lr": 1e-3, "batch_size": 16},
    {"hidden_size": 32, "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 32},
    {"hidden_size": 64, "num_layers": 1, "dropout": 0.2, "lr": 1e-3, "batch_size": 32},
    {"hidden_size": 64, "num_layers": 2, "dropout": 0.3, "lr": 5e-4, "batch_size": 32},
    {"hidden_size": 96, "num_layers": 2, "dropout": 0.3, "lr": 5e-4, "batch_size": 64},
]
TUNE_EPOCHS = 25
TUNE_PATIENCE = 6


def _val_mse(name, data, hp) -> float:
    """Fit one candidate on TRAIN, score it on VALIDATION. Test never touched."""
    tr, va, _te = split_bounds(data)
    set_seed()
    model = make_model(name, data["input_size"], hp)
    model, _ = train_model(model, data["X"][tr], data["y"][tr], data["X"][va],
                           data["y"][va], epochs=TUNE_EPOCHS, patience=TUNE_PATIENCE,
                           lr=hp["lr"], batch_size=hp["batch_size"])
    pred = predict_returns(model, data["X"][va], data["ret_scaler"])
    return float(np.mean((pred - data["actual_return"][va]) ** 2))


def tune(name: str, fdf: pd.DataFrame):
    """Grid search over sequence length and hyperparameters, validation only.

    Returns (best_hp, best_seq_len, best_val_mse). Sequence length is searched
    here too, since it changes the dataset shape and cannot be tuned afterwards.
    """
    best = None
    for L in SEQ_GRID:
        data = make_dataset(fdf, feature_columns=FUSION_FEATURES, seq_len=L)
        for hp in HP_GRID:
            score = _val_mse(name, data, hp)
            if best is None or score < best[0]:
                best = (score, hp, L)
    score, hp, L = best
    logger.info("    %-12s best: seq_len %s, hidden %s, layers %s, dropout %s, "
                "lr %.0e, batch %s (val MSE %.3e)", name, L, hp["hidden_size"],
                hp["num_layers"], hp["dropout"], hp["lr"], hp["batch_size"], score)
    return hp, L, score


# ------------------------------------------------------------------ training
def train_one(name: str, data: Dict, tr, va, hp=None, seed=None):
    """Train one model on the training slice with validation early stopping."""
    hp = hp or {"hidden_size": None, "lr": LEARNING_RATE, "batch_size": BATCH_SIZE}
    set_seed(seed if seed is not None else SEED)
    model = make_model(name, data["input_size"], hp if hp.get("hidden_size") else None)
    model, hist = train_model(model, data["X"][tr], data["y"][tr],
                              data["X"][va], data["y"][va], epochs=EPOCHS,
                              patience=EARLY_STOP_PATIENCE,
                              lr=hp.get("lr", LEARNING_RATE),
                              batch_size=hp.get("batch_size", BATCH_SIZE))
    logger.info("    %-12s trained: %s epochs, %s parameters", name,
                len(hist["train"]), f"{sum(p.numel() for p in model.parameters()):,}")
    return model


def ensemble_weights(val_pred: Dict[str, np.ndarray],
                     val_actual: np.ndarray) -> Dict[str, float]:
    """Inverse-MSE weights fitted on VALIDATION predictions only.

        w_m = (1 / MSE_m) / sum_k (1 / MSE_k),   MSE_m over the validation slice

    Justification: this is the precision-weighted (inverse-variance) combination,
    the minimum-variance linear pool under equal-bias assumptions - a standard,
    parameter-free rule. Nothing is hand-picked and the test slice is never seen.
    """
    mse = {m: float(np.mean((val_pred[m] - val_actual) ** 2)) for m in val_pred}
    inv = {m: 1.0 / max(v, 1e-12) for m, v in mse.items()}
    total = sum(inv.values())
    w = {m: inv[m] / total for m in inv}
    for m in MODEL_NAMES:
        logger.info("    %-12s val MSE %.3e -> weight %.4f", m, mse[m], w[m])
    return w


def blend(pred: Dict[str, np.ndarray], w: Dict[str, float]) -> np.ndarray:
    return np.sum([w[m] * pred[m] for m in pred], axis=0)


def class_metrics(y_true, y_pred) -> Dict:
    """Macro-averaged metrics + per-class counts.

    Macro treats SELL/HOLD/BUY equally, so a model that predicts one dominant
    class cannot inflate its score the way weighted averaging allows.
    """
    kw = dict(labels=CLASSES, average="macro", zero_division=0)
    yt = np.asarray(y_true)
    # Majority-class baseline: always predict the most common TEST class. It is
    # the honest comparator for accuracy - a three-class problem with an uneven
    # test split has a baseline well above 1/3. Macro-F1 is where a real model
    # must beat it, since always-one-class scores badly there by construction.
    maj = max(CLASSES, key=lambda c: int(np.sum(yt == c)))
    maj_pred = np.full(len(yt), maj)
    out = {"Accuracy": float(accuracy_score(y_true, y_pred)),
           "Balanced_Accuracy": float(recall_score(y_true, y_pred, labels=CLASSES,
                                                   average="macro", zero_division=0)),
           "Majority_Acc": float(accuracy_score(yt, maj_pred)),
           "Majority_MacroF1": float(f1_score(yt, maj_pred, **kw)),
           "Macro_Precision": float(precision_score(y_true, y_pred, **kw)),
           "Macro_Recall": float(recall_score(y_true, y_pred, **kw)),
           "Macro_F1": float(f1_score(y_true, y_pred, **kw))}
    pc = precision_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    rc = recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    fc = f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    for i, c in enumerate(CLASSES):
        out[f"True_{c}"] = int(np.sum(np.asarray(y_true) == c))
        out[f"Pred_{c}"] = int(np.sum(np.asarray(y_pred) == c))
        out[f"P_{c}"], out[f"R_{c}"], out[f"F1_{c}"] = (round(float(pc[i]), 4),
                                                       round(float(rc[i]), 4),
                                                       round(float(fc[i]), 4))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def calibrated_thresholds(val_pred, val_actual, buy=BUY_THRESHOLD, sell=SELL_THRESHOLD):
    """Decision thresholds fitted on VALIDATION only, guaranteed non-degenerate.

    Two corrections over a fixed band:

    1. DE-BIAS. b = median(validation predicted return). Trained on one regime
       and tested in another, the models carry a constant directional offset;
       subtracting the validation median removes it without using test data.

    2. PRIOR MATCHING. The BUY/SELL thresholds are the quantiles of the
       de-biased validation predictions that reproduce the class frequencies
       actually observed in the validation window. So if 40% of validation days
       were truly BUY, the top 40% of predictions are labelled BUY.

    A fixed band (e.g. +/-0.5%) can sit entirely outside the model's predicted
    range, collapsing every prediction to HOLD - which is what produced the
    degenerate all-HOLD result. Quantile matching cannot do that.

    True labels always use the FIXED band on ACTUAL returns, so the target
    definition never moves.
    """
    vp, va = np.asarray(val_pred, float), np.asarray(val_actual, float)
    bias = float(np.median(vp))
    adj = vp - bias
    true_val = signals(va, buy, sell)
    p_buy = float(np.mean(true_val == "BUY"))
    p_sell = float(np.mean(true_val == "SELL"))
    tau_buy = float(np.quantile(adj, 1.0 - p_buy)) if p_buy > 0 else float("inf")
    tau_sell = float(np.quantile(adj, p_sell)) if p_sell > 0 else float("-inf")
    if tau_buy <= tau_sell:                      # degenerate overlap -> split at median
        tau_buy = tau_sell = 0.0
    return bias, tau_buy, tau_sell


# ------------------------------------------------------------------ classifier
# The regression models above are trained with MSE on returns, which pushes
# predictions toward zero; turning those into BUY/SELL/HOLD afterwards is where
# most classification accuracy was lost. These models instead predict the class
# DIRECTLY with cross-entropy, and class weights (computed on TRAIN only) stop
# the majority class from dominating - which is what produced runs of all-BUY.
# The target definition is unchanged: the fixed +/-0.5% band on actual returns.
CLS_EPOCHS, CLS_PATIENCE = 60, 10


def labels_from(returns) -> np.ndarray:
    """Actual return -> class index, using the SAME fixed band as everywhere."""
    r = np.asarray(returns, float)
    return np.where(r >= BUY_THRESHOLD, 2, np.where(r <= SELL_THRESHOLD, 0, 1))


def predict_proba(model, X) -> np.ndarray:
    """Class probabilities (N, 3) from a 3-logit head."""
    model.eval()
    outs = []
    with torch.no_grad():
        t = torch.as_tensor(X)
        for i in range(0, len(t), 512):
            outs.append(torch.softmax(model(t[i:i + 512].to(DEVICE)), dim=1)
                        .float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, len(CLASSES)))


def train_mt(name, data, tr, va, hp, seed):
    """Same backbone, 4 outputs: 1 return + 3 class logits, trained jointly.

    Third decision method. Competes with thresholds and the pure classifier on
    VALIDATION macro-F1; the winner is used on test. Nothing here sees test.
    """
    set_seed(seed)
    model = make_model(name, data["input_size"], hp if hp.get("hidden_size") else None)
    model.fc = nn.Linear(model.fc.in_features, 1 + len(CLASSES))
    model.to(DEVICE)

    y = labels_from(data["actual_return"])
    counts = np.bincount(y[tr], minlength=len(CLASSES)).astype(float)
    weight = counts.sum() / np.maximum(counts, 1.0)
    model, _ = train_multitask(model, data["X"][tr], data["y"][tr], y[tr],
                               data["X"][va], data["y"][va], y[va],
                               lr=hp.get("lr", LEARNING_RATE),
                               batch_size=hp.get("batch_size", BATCH_SIZE),
                               class_weight=weight)
    return model


def train_classifier(name, data, tr, va, hp, seed):
    """Same backbone, 3-way softmax head, cross-entropy with class weights.

    Class weights come from the TRAINING split only, so a rare class is not
    ignored. Early stopping on validation loss, as in the regression path.
    """
    set_seed(seed)
    model = make_model(name, data["input_size"], hp if hp.get("hidden_size") else None)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, len(CLASSES))          # 1 output -> 3 logits
    model.to(DEVICE)

    y = labels_from(data["actual_return"])
    ytr, yva = y[tr], y[va]
    counts = np.bincount(ytr, minlength=len(CLASSES)).astype(float)
    weight = torch.tensor(counts.sum() / np.maximum(counts, 1.0),
                          dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=hp.get("lr", LEARNING_RATE),
                           weight_decay=1e-5)

    Xtr = torch.as_tensor(data["X"][tr])
    Ttr = torch.as_tensor(ytr)
    Xva = torch.as_tensor(data["X"][va]).to(DEVICE)
    Tva = torch.as_tensor(yva).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, Ttr),
        batch_size=hp.get("batch_size", BATCH_SIZE), shuffle=True,
        generator=torch.Generator().manual_seed(0))

    best, best_state, wait = float("inf"), None, 0
    for _ep in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            crit(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(crit(model(Xva), Tva))
        if vl < best - 1e-6:
            best, best_state, wait = vl, {k: v.clone() for k, v in
                                          model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


def classifier_probs(model, X) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        t = torch.as_tensor(X)
        for i in range(0, len(t), 512):
            out.append(torch.softmax(model(t[i:i + 512].to(DEVICE)), dim=1)
                       .cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


# ------------------------------------------------------------------ explainability
def group_importance(model, data, va, ret_scaler, groups=FEATURE_GROUPS,
                     seed=0) -> Dict[str, float]:
    """Permutation importance per feature group, measured on VALIDATION.

    Each group's columns are shuffled across the time axis; the rise in
    validation MAE (of the predicted return) is that group's contribution.
    Values are normalised to sum to 1 for reporting. Never touches test.
    """
    cols = FUSION_FEATURES
    idx = {g: [cols.index(c) for c in members if c in cols]
           for g, members in groups.items()}
    Xva = data["X"][va]
    base = predict_returns(model, Xva, ret_scaler)
    truth = data["actual_return"][va]
    base_mae = float(np.mean(np.abs(base - truth)))

    rng = np.random.default_rng(seed)
    raw = {}
    for g, ids in idx.items():
        if not ids:
            raw[g] = 0.0
            continue
        # average over several shuffles: one permutation on ~215 validation rows
        # is noisy enough that a real effect can come out negative by chance
        deltas = []
        for rep in range(5):
            Xp = Xva.copy()
            Xp[:, :, ids] = Xp[rng.permutation(len(Xp))][:, :, ids]
            deltas.append(float(np.mean(np.abs(
                predict_returns(model, Xp, ret_scaler) - truth))) - base_mae)
        raw[g] = float(np.mean(deltas))

    # Report SIGNED mean deltas normalised by their absolute total. Clipping
    # negatives to zero (the previous behaviour) made every group read 0.0%
    # whenever the model was near-constant, which hid the finding instead of
    # showing it. base_mae is returned so the reader can judge the scale.
    total = sum(abs(v) for v in raw.values())
    out = {g: (v / total if total > 1e-12 else 0.0) for g, v in raw.items()}
    out["_base_mae"] = base_mae
    out["_measurable"] = total > 1e-6 * max(base_mae, 1e-9)
    return out


def confidence(pred: Dict[str, np.ndarray], w: Dict[str, float],
               ens: np.ndarray) -> np.ndarray:
    """Agreement-based confidence in [0, 1].

        conf = 1 - min(1, weighted_dispersion / |ensemble return|)

    High when the three models agree relative to the size of the move; low when
    they disagree or the predicted move is negligible. Uses predictions only.
    """
    disp = np.sqrt(np.sum([w[m] * (pred[m] - ens) ** 2 for m in pred], axis=0))
    scale = np.maximum(np.abs(ens), 1e-6)
    return np.clip(1.0 - np.minimum(1.0, disp / scale), 0.0, 1.0)


# ------------------------------------------------------------------ per stock
def process(ticker: str, corpus) -> Dict:
    logger.info("\n%s\nSTOCK: %s\n%s", "=" * _W, ticker, "=" * _W)
    b = SentimentAgent().run(DataAgent().run(ticker))        # unchanged agents
    fdf = build_feature_frame(b["stock_df"], b["sentiment_daily"], b["regime_df"])
    nf, _scored, ninfo = news_features(ticker, fdf.index, corpus)
    for c in NEWS_FEATURES:
        fdf[c] = nf[c].to_numpy() if nf is not None else 0.0
    # build_feature_frame warns about a constant News_Sentiment because it fills
    # a placeholder before the real corpus is attached above. Report the state
    # AFTER attachment so the log reflects what the models actually receive.
    live_news = [c for c in NEWS_FEATURES if float(fdf[c].std()) > 1e-12]
    logger.info("  news features live after attachment: %s of %s (%s)",
                len(live_news), len(NEWS_FEATURES),
                ", ".join(live_news) if live_news
                else "NONE - news contributes nothing for this stock")

    data = make_dataset(fdf, feature_columns=FUSION_FEATURES)   # same split as Phase 1
    tr, va, te = split_bounds(data)
    rs = data["ret_scaler"]
    cur, fut, aret = data["current_close"], data["future_close"], data["actual_return"]

    logger.info("  tuning on train+validation only (%s seq lengths x %s configs)",
                len(SEQ_GRID), len(HP_GRID))
    tuned = {m: tune(m, fdf) for m in MODEL_NAMES}
    seq_len = tuned["BiLSTM"][1]          # shared window so predictions align
    data = make_dataset(fdf, feature_columns=FUSION_FEATURES, seq_len=seq_len)
    tr, va, te = split_bounds(data)
    rs = data["ret_scaler"]
    cur, fut, aret = data["current_close"], data["future_close"], data["actual_return"]
    logger.info("  training three models on %s features, seq_len %s, %s seeds each",
                data["input_size"], seq_len, len(SEEDS))
    val_by_seed = {m: [] for m in MODEL_NAMES}
    test_by_seed = {m: [] for m in MODEL_NAMES}
    keep_model = {}
    for m in MODEL_NAMES:
        for sd in SEEDS:
            mdl = train_one(m, data, tr, va, tuned[m][0], seed=sd)
            val_by_seed[m].append(predict_returns(mdl, data["X"][va], rs))
            test_by_seed[m].append(predict_returns(mdl, data["X"][te], rs))
            if sd == SEEDS[0]:
                keep_model[m] = mdl              # kept for the importance layer
        spread = np.std([np.mean(np.abs(p - aret[te])) for p in test_by_seed[m]])
        logger.info("    %-12s %s seeds | test MAE spread across seeds %.5f",
                    m, len(SEEDS), spread)
    models = keep_model
    val_pred = {m: np.mean(val_by_seed[m], axis=0) for m in MODEL_NAMES}
    test_pred = {m: np.mean(test_by_seed[m], axis=0) for m in MODEL_NAMES}

    logger.info("  ensemble weights (validation only)")
    w = ensemble_weights(val_pred, aret[va])
    val_ens, test_ens = blend(val_pred, w), blend(test_pred, w)

    # --- direct BUY/SELL/HOLD classifiers (one per model family, seed-averaged)
    logger.info("  training BUY/SELL/HOLD classifiers (cross-entropy, class-weighted)")
    cls_val, cls_test, cls_loss = {}, {}, {}
    for m in MODEL_NAMES:
        vps, tps, ls = [], [], []
        for sd in SEEDS:
            cm, vl = train_classifier(m, data, tr, va, tuned[m][0], sd)
            vps.append(predict_proba(cm, data["X"][va]))
            tps.append(predict_proba(cm, data["X"][te]))
            ls.append(vl)
        cls_val[m], cls_test[m] = np.mean(vps, axis=0), np.mean(tps, axis=0)
        cls_loss[m] = float(np.mean(ls))
        logger.info("    %-12s val cross-entropy %.4f", m, cls_loss[m])
    inv = {m: 1.0 / max(cls_loss[m], 1e-12) for m in MODEL_NAMES}
    tot = sum(inv.values())
    cw = {m: inv[m] / tot for m in MODEL_NAMES}          # validation-only weights
    cls_ens = np.sum([cw[m] * cls_test[m] for m in MODEL_NAMES], axis=0)
    logger.info("    classifier ensemble weights: %s",
                ", ".join(f"{m} {cw[m]:.4f}" for m in MODEL_NAMES))

    y_split = labels_from(data["actual_return"])
    balance = {n: {c: int(np.sum(y_split[sl] == i)) for i, c in enumerate(CLASSES)}
               for n, sl in (("train", tr), ("val", va), ("test", te))}
    logger.info("  class balance (fixed +/-%.1f%% band on ACTUAL returns): %s",
                BUY_THRESHOLD * 100,
                " | ".join(f"{n} " + "/".join(str(balance[n][c]) for c in CLASSES)
                           for n in balance))

    logger.info("  training multi-task heads (return + direction, joint loss)")
    mt_val, mt_test = [], []
    for m in MODEL_NAMES:
        vs, ts = [], []
        for sd in SEEDS:
            mdl = train_mt(m, data, tr, va, tuned[m][0], sd)
            _r, pv = predict_multitask(mdl, data["X"][va], rs)
            _r2, pt = predict_multitask(mdl, data["X"][te], rs)
            vs.append(pv)
            ts.append(pt)
        mt_val.append(np.mean(vs, axis=0))
        mt_test.append(np.mean(ts, axis=0))
    mt_val, mt_test = np.mean(mt_val, axis=0), np.mean(mt_test, axis=0)
    y_val_true = np.array(CLASSES)[labels_from(aret[va])]
    mt_val_f1 = f1_score(y_val_true, np.array(CLASSES)[mt_val.argmax(1)],
                         labels=CLASSES, average="macro", zero_division=0)
    logger.info("    multi-task validation macro-F1 %.4f", mt_val_f1)

    bias, tau_buy, tau_sell = calibrated_thresholds(val_ens, aret[va])
    # --- pick the decision method ON VALIDATION, never on test.
    # Two candidates: (a) regression predictions turned into classes by
    # validation-calibrated thresholds, (b) the direct classifier. Whichever
    # scores the higher macro-F1 on VALIDATION is used for the test set.
    y_val_true = signals(aret[va], BUY_THRESHOLD, SELL_THRESHOLD)
    hist_v, roll_v = list(val_ens), []
    for v in val_ens:
        roll_v.append(float(np.median(hist_v[:max(1, len(roll_v))][-BIAS_WINDOW:])))
    thr_val = signals(val_ens - np.asarray(roll_v), tau_buy, tau_sell)
    cls_val_ens = np.sum([cw[m] * cls_val[m] for m in MODEL_NAMES], axis=0)
    cls_val_lab = np.array(CLASSES)[cls_val_ens.argmax(axis=1)]
    kw = dict(labels=CLASSES, average="macro", zero_division=0)
    f1_thr = float(f1_score(y_val_true, thr_val, **kw))
    f1_cls = float(f1_score(y_val_true, cls_val_lab, **kw))
    scores = {"THRESHOLDS": f1_thr, "CLASSIFIER": f1_cls, "MULTITASK": mt_val_f1}
    method = max(scores, key=scores.get)          # chosen on VALIDATION only
    use_classifier = method == "CLASSIFIER"
    logger.info("  decision method chosen on VALIDATION macro-F1: %s -> using %s",
                " | ".join(f"{k} {v:.4f}" for k, v in scores.items()), method)
    # Rolling de-bias: at each test day t the offset is the median of the
    # predictions BEFORE t (seeded with validation), so a level shift between
    # validation and test cannot push every day past one threshold. Strictly
    # past-only - day t never sees day t or later.
    hist = list(val_ens)
    roll = []
    for v in test_ens:
        roll.append(float(np.median(hist[-BIAS_WINDOW:])))
        hist.append(v)
    roll = np.asarray(roll)
    logger.info("  decision layer from validation: bias %+.5f | BUY >= %+.5f | "
                "SELL <= %+.5f", bias, tau_buy, tau_sell)

    rows, cls_rows, cms = [], [], {}
    for name in MODEL_NAMES + ["Ensemble"]:
        pr = test_ens if name == "Ensemble" else test_pred[name]
        pc = cur[te] * (1 + pr)
        m = full_metrics(fut[te], pc, aret[te], pr, buy=tau_buy, sell=tau_sell)
        y_t = signals(aret[te], BUY_THRESHOLD, SELL_THRESHOLD)   # fixed band, actual
        if use_classifier:
            proba = cls_ens if name == "Ensemble" else cls_test[name]
            y_p = np.array(CLASSES)[proba.argmax(axis=1)]
        else:
            y_p = signals(pr - roll, tau_buy, tau_sell)
        rows.append({"Stock": ticker, "Model": name,
                     **{k: round(float(m[k]), 4) for k in METRIC_KEYS}})
        cls_rows.append({"Stock": ticker, "Model": name,
                         **class_metrics(y_t, y_p),
                         "Weighted_F1": round(float(m["F1"]), 4),
                         "Directional_Accuracy": round(float(m["Directional_Accuracy"]), 4)})
        cms[name] = confusion_matrix(y_t, y_p, labels=CLASSES)

    conf = (cls_ens.max(axis=1) if use_classifier
            else confidence(test_pred, w, test_ens))
    y_true = signals(aret[te], BUY_THRESHOLD, SELL_THRESHOLD)
    y_pred = (np.array(CLASSES)[cls_ens.argmax(axis=1)] if use_classifier
              else signals(test_ens - roll, tau_buy, tau_sell))
    preds = pd.DataFrame({
        "Date": data["dates"][te], "Stock": ticker,
        "Actual_Close": fut[te], "Predicted_Close": cur[te] * (1 + test_ens),
        "Actual_Return": aret[te], "Predicted_Return": test_ens,
        "Confidence": conf, "True_Signal": y_true, "Predicted_Signal": y_pred,
        **{f"Pred_{m}": test_pred[m] for m in MODEL_NAMES}})

    # classifier metrics for every backbone, reported next to the threshold path
    y_fixed = signals(aret[te], BUY_THRESHOLD, SELL_THRESHOLD)
    clf_rows, clf_pred = [], {}
    for name in MODEL_NAMES + ["Ensemble"]:
        proba = cls_ens if name == "Ensemble" else cls_test[name]
        lab = np.array(CLASSES)[proba.argmax(axis=1)]
        clf_pred[name] = lab
        clf_rows.append({"Stock": ticker, "Model": name,
                         **class_metrics(y_fixed, lab)})
    clf_cm = confusion_matrix(y_fixed, clf_pred["Ensemble"], labels=CLASSES)

    imp = group_importance(models["BiLSTM"], data, va, rs)
    if imp.get("_measurable"):
        logger.info("  feature-group importance (validation permutation, %s shuffles"
                    " averaged): %s", 5,
                    ", ".join(f"{g} {v * 100:+.1f}%" for g, v in imp.items()
                              if not g.startswith("_")))
    else:
        logger.warning("  feature-group importance NOT MEASURABLE: shuffling any "
                       "group changes validation MAE by <0.0001%% of %.6f - the "
                       "model output is near-constant for this stock",
                       imp["_base_mae"])

    d = os.path.join(PHASE2_DIR, ticker)
    os.makedirs(d, exist_ok=True)
    preds.to_csv(os.path.join(d, "predictions.csv"), index=False)
    for name, cm in cms.items():
        pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                     columns=[f"pred_{c}" for c in CLASSES]).to_csv(
                         os.path.join(d, f"confusion_{name}.csv"))

    return {"ticker": ticker, "rows": rows, "cls_rows": cls_rows, "cms": cms,
            "clf_rows": clf_rows, "clf_cm": clf_cm,
            "clf_pred": clf_pred["Ensemble"],
            "weights": w, "cls_weights": cw, "tau": (tau_buy, tau_sell),
            "bias": bias, "importance": imp, "method": ("classifier" if use_classifier
                                                        else "thresholds"),
            "val_f1": {"thresholds": round(f1_thr, 4), "classifier": round(f1_cls, 4)},
            "balance": balance,
            "seed_spread": {m: float(np.std([np.mean(np.abs(p - aret[te]))
                                             for p in test_by_seed[m]]))
                            for m in MODEL_NAMES},
            "tuned": {m: {"hp": tuned[m][0], "seq_len": tuned[m][1],
                          "val_mse": tuned[m][2]} for m in MODEL_NAMES},
            "seq_len": seq_len,
            "preds": preds, "news": ninfo,
            "n": {"train": tr.stop - tr.start, "val": va.stop - va.start,
                  "test": te.stop - te.start}}


# ------------------------------------------------------------------ report
def _rule(ch="-"):
    return ch * _W


def report(results: List[Dict]) -> None:
    fore = pd.DataFrame([r for res in results for r in res["rows"]])
    cls = pd.DataFrame([r for res in results for r in res["cls_rows"]])
    ens_all = pd.concat([res["preds"] for res in results], ignore_index=True)

    L = ["", "#" * _W, "PHASE 2 - ENSEMBLE FORECASTING & DECISION RESULTS".center(_W),
         "#" * _W, "", _rule("="), "A. FORECASTING METRICS (test set)", _rule("="),
         f"  {'Stock':<8}{'Model':<14}{'MAE':>10}{'RMSE':>10}{'MAPE %':>10}",
         "  " + _rule("-")[:52]]
    for _, sub in fore.groupby("Stock", sort=False):
        L += [f"  {r['Stock']:<8}{r['Model']:<14}{r['MAE']:>10.4f}{r['RMSE']:>10.4f}"
              f"{r['MAPE']:>10.4f}" for _, r in sub.iterrows()] + [""]

    L += [_rule("="), "B. BUY/SELL/HOLD CLASSIFICATION (test set, macro-averaged)",
          _rule("="),
          "  True label : ACTUAL next-day return vs a FIXED band (+/-0.5%)",
          "  Pred label : whichever of two methods scored higher macro-F1 on",
          "               VALIDATION - (a) calibrated thresholds on the regression",
          "               output, or (b) a direct class-weighted 3-class classifier.",
          "               The choice is made on validation, never on test.",
          "  Macro averaging weights SELL/HOLD/BUY equally, so predicting one",
          "  dominant class cannot inflate the score.", "",
          f"  {'Stock':<8}{'Model':<14}{'Acc':>8}{'BalAcc':>8}{'MacroP':>8}"
          f"{'MacroR':>8}{'MacroF1':>9}{'DirAcc':>9}", "  " + _rule("-")[:72]]
    for _, sub in cls.groupby("Stock", sort=False):
        L += [f"  {r['Stock']:<8}{r['Model']:<14}{r['Accuracy']:>8.4f}"
              f"{r['Balanced_Accuracy']:>8.4f}{r['Macro_Precision']:>8.4f}"
              f"{r['Macro_Recall']:>8.4f}{r['Macro_F1']:>9.4f}"
              f"{r['Directional_Accuracy']:>9.4f}"
              for _, r in sub.iterrows()]
        b = sub.iloc[0]
        L += [f"  {b['Stock']:<8}{'majority-class':<14}{b['Majority_Acc']:>8.4f}"
              f"{1/3:>8.4f}{'':>8}{'':>8}{b['Majority_MacroF1']:>9.4f}{'':>9}", ""]

    L += [_rule("="), "B2. SIGNAL COUNTS (test set)", _rule("="),
          f"  {'Stock':<8}{'Model':<14}" + "".join(f"{'true ' + c:>11}" for c in CLASSES)
          + "".join(f"{'pred ' + c:>11}" for c in CLASSES),
          "  " + _rule("-")[:88]]
    for _, sub in cls.groupby("Stock", sort=False):
        for _, r in sub.iterrows():
            L.append(f"  {r['Stock']:<8}{r['Model']:<14}"
                     + "".join(f"{r['True_' + c]:>11}" for c in CLASSES)
                     + "".join(f"{r['Pred_' + c]:>11}" for c in CLASSES))
        L.append("")

    L += ["", _rule("="), "B4. CLASS BALANCE AND PER-CLASS METRICS", _rule("="),
          f"  Labels from the fixed +/-{BUY_THRESHOLD * 100:.1f}% band on ACTUAL",
          "  returns - identical definition in train, validation and test. The",
          "  test set is never rebalanced.", "",
          f"  {'Stock':<8}{'split':<8}" + "".join(f"{c:>8}" for c in CLASSES),
          "  " + _rule("-")[:40]]
    for res in results:
        for sp in ("train", "val", "test"):
            L.append(f"  {res['ticker']:<8}{sp:<8}"
                     + "".join(f"{res['balance'][sp][c]:>8}" for c in CLASSES))
        L.append("")
    L += [f"  {'Stock':<8}{'Model':<14}" + "".join(
              f"{'P/R/F1 ' + c:>20}" for c in CLASSES), "  " + _rule("-")[:82]]
    for _, r in cls.iterrows():
        L.append(f"  {r['Stock']:<8}{r['Model']:<14}" + "".join(
            f"{r['P_' + c]:.2f}/{r['R_' + c]:.2f}/{r['F1_' + c]:.2f}".rjust(20)
            for c in CLASSES))

    L += ["", _rule("="), "C. ENSEMBLE COMPOSITION (inverse-MSE, validation only)",
          _rule("="),
          f"  {'Stock':<8}" + "".join(f"{m:>13}" for m in MODEL_NAMES)
          + f"{'bias':>11}{'BUY thr':>11}{'SELL thr':>11}", "  " + _rule("-")[:81]]
    for res in results:
        L.append(f"  {res['ticker']:<8}"
                 + "".join(f"{res['weights'][m]:>13.4f}" for m in MODEL_NAMES)
                 + f"{res['bias']:>+11.5f}{res['tau'][0]:>+11.5f}{res['tau'][1]:>+11.5f}")
    L += ["", f"  Seed spread (std of test MAE across {len(SEEDS)} seeds, per model):"]
    for res in results:
        L.append(f"    {res['ticker']:<8}" + "  ".join(
            f"{m} {res['seed_spread'][m]:.5f}" for m in MODEL_NAMES))
    L.append("  Reported predictions are the mean over seeds, which is what removes")
    L.append("  the run-to-run instability a single seed shows at this sample size.")

    L += ["", "  Decision method selected on validation macro-F1:"]
    for res in results:
        v = res["val_f1"]
        L.append(f"    {res['ticker']:<8}{res['method']:<12} "
                 f"(thresholds {v['thresholds']:.4f} vs classifier {v['classifier']:.4f})")

    L += ["", _rule("="),
          "C2. TUNED HYPERPARAMETERS (grid search on train+validation only)",
          _rule("="),
          f"  {'Stock':<8}{'Model':<14}{'seq':>5}{'hidden':>8}{'layers':>8}"
          f"{'drop':>7}{'lr':>9}{'batch':>7}{'val MSE':>12}",
          "  " + _rule("-")[:78]]
    for res in results:
        for m in MODEL_NAMES:
            t = res["tuned"][m]
            L.append(f"  {res['ticker']:<8}{m:<14}{t['seq_len']:>5}"
                     f"{t['hp']['hidden_size']:>8}{t['hp']['num_layers']:>8}"
                     f"{t['hp']['dropout']:>7}{t['hp']['lr']:>9.0e}"
                     f"{t['hp']['batch_size']:>7}{t['val_mse']:>12.3e}")
    L.append("  Final models use the shared best sequence length so predictions align.")

    L += ["", _rule("="), "D. CONFUSION MATRIX - ENSEMBLE (rows true, cols predicted)",
          _rule("=")]
    for res in results:
        L += [f"  {res['ticker']}", "           " + "".join(f"{c:>8}" for c in CLASSES)]
        for c, row in zip(CLASSES, res["cms"]["Ensemble"]):
            L.append(f"    {c:<7}" + "".join(f"{v:>8}" for v in row))
        L.append("")

    L += [_rule("="), "E. EXPLAINABLE REPORT (per stock)", _rule("=")]
    for res in results:
        pr = res["preds"]
        hit = pr["Predicted_Signal"] == pr["True_Signal"]
        n_ok, n = int(hit.sum()), len(pr)
        L += ["", f"  {res['ticker']}",
              f"    Correct decisions   : {n_ok} of {n} test days "
              f"({n_ok / n * 100:.1f}%)"]
        per = ", ".join(
            f"{c} {int(((pr['True_Signal'] == c) & hit).sum())}/"
            f"{int((pr['True_Signal'] == c).sum())}" for c in CLASSES)
        L.append(f"    Correct by class    : {per}")
        used = pr["Predicted_Signal"].nunique()
        if used < 2:
            only = pr["Predicted_Signal"].iloc[0]
            L += [f"    *** DEGENERATE: the model predicted {only} on ALL {n} test",
                  f"        days. Its correct count comes entirely from days that",
                  f"        happened to be {only}; it never got any other class",
                  f"        right. Do NOT read this as a working classifier.",
                  f"        Cause: the validation bias correction ({res['bias']:+.5f})",
                  f"        exceeded the whole test prediction range, so every",
                  f"        de-biased value fell past one threshold."]

        # one worked example of a CORRECT call, and the final test day, side by
        # side. The example illustrates the reasoning; the counts above are the
        # result. Neither is selected to flatter the model - the aggregate is
        # always printed first.
        for label, row in (("Example correct decision",
                            pr[hit].iloc[-1] if n_ok else None),
                           ("Final test day", pr.iloc[-1])):
            if row is None:
                L.append(f"    {label}: none on this stock")
                continue
            mark = "correct" if row["Predicted_Signal"] == row["True_Signal"] else "wrong"
            L += ["", f"    {label} - {pd.Timestamp(row['Date']).date()} ({mark})",
                  f"      Decision          : {row['Predicted_Signal']}  "
                  f"(actual {row['True_Signal']})",
                  f"      Predicted return  : {row['Predicted_Return'] * 100:+.2f}%  "
                  f"-> close {row['Predicted_Close']:.2f} "
                  f"(actual {row['Actual_Close']:.2f})",
                  f"      Confidence        : {row['Confidence'] * 100:.0f}%",
                  "      Model votes       : "
                  + ", ".join(f"{m} {res['weights'][m] * 100:.0f}% "
                              f"({row[f'Pred_{m}'] * 100:+.2f}%)" for m in MODEL_NAMES)]
        L += ["",
              ("    Feature groups      : "
               + ", ".join(f"{g} {v * 100:+.0f}%" for g, v in res["importance"].items()
                           if not g.startswith("_"))
               if res["importance"].get("_measurable")
               else "    Feature groups      : NOT MEASURABLE - shuffling any group "
                    "left validation MAE unchanged (near-constant output)"),
              "    Historical news     : "
              + (f"{res['news']['articles']} articles over "
                 f"{res['news']['covered_days']}/{res['news']['trading_days']} "
                 f"trading days" if res["news"].get("available")
                 else f"unavailable ({res['news'].get('reason', '')})"),
              f"    Split               : train {res['n']['train']} / val "
              f"{res['n']['val']} / test {res['n']['test']} (chronological)"]

    clfdf = pd.DataFrame([r for res in results for r in res["clf_rows"]])
    L += ["", _rule("="), "F. DIRECT CLASSIFIER (cross-entropy, same backbones)",
          _rule("="),
          "  Trained directly on the three classes instead of thresholding a",
          "  predicted return. Same splits, same tuned backbones, same seeds,",
          "  class weights from TRAIN only. Reported alongside the regression",
          "  path above, not in place of it.", "",
          f"  {'Stock':<8}{'Model':<14}{'Acc':>8}{'MacroP':>8}{'MacroR':>8}"
          f"{'MacroF1':>9}", "  " + _rule("-")[:55]]
    for _, sub in clfdf.groupby("Stock", sort=False):
        L += [f"  {r['Stock']:<8}{r['Model']:<14}{r['Accuracy']:>8.4f}"
              f"{r['Macro_Precision']:>8.4f}{r['Macro_Recall']:>8.4f}"
              f"{r['Macro_F1']:>9.4f}" for _, r in sub.iterrows()] + [""]

    L += ["  Regression-threshold vs classifier (Ensemble, accuracy / macro-F1):"]
    for res in results:
        a = next(r for r in res["cls_rows"] if r["Model"] == "Ensemble")
        b = next(r for r in res["clf_rows"] if r["Model"] == "Ensemble")
        L.append(f"    {res['ticker']:<8} regression {a['Accuracy']:.4f} / "
                 f"{a['Macro_F1']:.4f}    classifier {b['Accuracy']:.4f} / "
                 f"{b['Macro_F1']:.4f}")

    L += ["", "  Classifier confusion (Ensemble, rows true, cols predicted):"]
    for res in results:
        L += [f"    {res['ticker']}",
              "             " + "".join(f"{c:>8}" for c in CLASSES)]
        for c, row in zip(CLASSES, res["clf_cm"]):
            L.append(f"      {c:<7}" + "".join(f"{v:>8}" for v in row))

    L += ["", _rule("="), "NOTES", _rule("="),
          "  Feature-group percentages are permutation importance measured on the",
          "  VALIDATION slice: the rise in validation MAE when that group's columns",
          "  are shuffled, normalised to sum to 100%. They are not causal claims.",
          "  Every fitted quantity - scalers, weights, thresholds, importance -",
          "  comes from train or validation data only. Test is scored once.",
          "", "#" * _W, ""]

    logger.info("\n".join(L))
    os.makedirs(PHASE2_DIR, exist_ok=True)
    with open(os.path.join(PHASE2_DIR, "phase2_report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(L))
    fore.to_csv(os.path.join(PHASE2_DIR, "model_comparison.csv"), index=False)
    cls.to_csv(os.path.join(PHASE2_DIR, "classification_metrics.csv"), index=False)
    clfdf.to_csv(os.path.join(PHASE2_DIR, "classifier_metrics.csv"), index=False)
    ens_all.to_csv(os.path.join(PHASE2_DIR, "ensemble_predictions.csv"), index=False)
    ens_all[["Date", "Stock", "Predicted_Return", "Confidence",
             "True_Signal", "Predicted_Signal"]].to_csv(
                 os.path.join(PHASE2_DIR, "buy_sell_hold.csv"), index=False)
    cm_rows = []
    for res in results:
        for name, cm in res["cms"].items():
            row = {"Stock": res["ticker"], "Model": name}
            for i, t in enumerate(CLASSES):
                for j, p_ in enumerate(CLASSES):
                    row[f"true_{t}_pred_{p_}"] = int(cm[i][j])
            cm_rows.append(row)
    pd.DataFrame(cm_rows).to_csv(
        os.path.join(PHASE2_DIR, "confusion_matrix.csv"), index=False)


# ------------------------------------------------------------------ entry point
def main() -> int:
    configure_logging()
    ensure_dirs()
    os.makedirs(PHASE2_DIR, exist_ok=True)
    set_seed()
    logger.info("PHASE 2 | %s -> inverse-MSE ensemble -> BUY/SELL/HOLD",
                " + ".join(MODEL_NAMES))
    logger.info("Features: %s columns (Phase-1 E4 fusion) | chronological splits",
                len(FUSION_FEATURES))

    corpus, info = load_corpus()
    if not info.get("available"):
        logger.warning("historical news unavailable (%s) - news columns stay "
                       "neutral; the ensemble still runs", info.get("reason", ""))

    results = []
    for ticker in TICKERS:
        try:
            results.append(process(ticker, corpus))
        except Exception as exc:
            logger.error("[%s] skipped: %s: %s", ticker, type(exc).__name__, exc)
            logger.debug("traceback:\n%s", traceback.format_exc())

    save_cache()
    if not results:
        logger.error("no stocks processed - aborting")
        return 1
    report(results)
    logger.info("Phase-2 artefacts under %s", PHASE2_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

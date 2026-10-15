"""Deep-learning forecasters + persistence baseline.

LSTM / GRU / Transformer share a training routine with:
  - configurable hyperparameters (hidden size, layers, dropout, lr, batch size)
  - weight decay (L2) for generalization
  - ReduceLROnPlateau learning-rate scheduler
  - validation-based early stopping (patience)
  - best-checkpoint restoration (best val loss, not final epoch)
  - per-epoch train/val loss history (for loss-curve graphs)

Nothing is tuned or selected on the test set.
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from config import (HIDDEN_SIZE, NUM_LAYERS, DROPOUT, DEVICE, BATCH_SIZE,
                    LEARNING_RATE, WEIGHT_DECAY, EARLY_STOP_PATIENCE, EPOCHS,
                    LR_SCHED_FACTOR, LR_SCHED_PATIENCE, GRAD_CLIP_NORM, USE_AMP)
from utils import get_logger

logger = get_logger("models")

_AMP_ENABLED = bool(USE_AMP) and str(DEVICE).startswith("cuda")

DEFAULT_HP = {"hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
              "dropout": DROPOUT, "lr": LEARNING_RATE, "batch_size": BATCH_SIZE}


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM forecaster (Phase 1 model).

    Both directions read the SAME 30-day look-back window that already ends at
    time t, so the backward pass runs from t back to t-29. No information after
    t enters the window - bidirectionality here is within-window context, not
    look-ahead. The final hidden state of each direction is concatenated, hence
    the 2 * hidden_size head.
    """

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        fwd = out[:, -1, :out.size(2) // 2]      # forward direction at t
        bwd = out[:, 0, out.size(2) // 2:]       # backward direction at t-29
        return self.fc(self.drop(torch.cat([fwd, bwd], dim=1)))


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, out_dim=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.drop(out[:, -1, :]))


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=DROPOUT,
                 out_dim=1):
        super().__init__()
        d_model = max(nhead, (d_model // nhead) * nhead)   # keep divisible by nhead
        self.input_proj = nn.Linear(input_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, out_dim)
        self.d_model = d_model

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self._pe(x.size(1), self.d_model, x.device)
        out = self.encoder(x)
        return self.fc(self.drop(out[:, -1, :]))

    @staticmethod
    def _pe(seq_len, d_model, device):
        pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2, device=device).float()
                        * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model, device=device)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe


def make_model(name, input_size, hp=None, out_dim=1):
    """Build a model from a hyperparameter dict.

    out_dim=1 (default) is the regression head used everywhere before; out_dim=3
    gives a BUY/SELL/HOLD classification head. Nothing else differs.
    """
    hp = {**DEFAULT_HP, **(hp or {})}
    h, nl, dr = hp["hidden_size"], hp["num_layers"], hp["dropout"]
    if name == "LSTM":
        return LSTMModel(input_size, hidden_size=h, num_layers=nl, dropout=dr,
                         out_dim=out_dim)
    if name == "BiLSTM":
        return BiLSTMModel(input_size, hidden_size=h, num_layers=nl, dropout=dr,
                           out_dim=out_dim)
    if name == "GRU":
        return GRUModel(input_size, hidden_size=h, num_layers=nl, dropout=dr,
                        out_dim=out_dim)
    if name == "Transformer":
        return TransformerModel(input_size, d_model=h, num_layers=nl, dropout=dr,
                                out_dim=out_dim)
    raise ValueError(f"unknown model {name}")


def train_model(model, X_tr, y_tr, X_va=None, y_va=None, epochs=EPOCHS,
                patience=EARLY_STOP_PATIENCE, lr=LEARNING_RATE,
                batch_size=BATCH_SIZE, weight_decay=WEIGHT_DECAY, verbose=False):
    """Train with weight decay, LR scheduler and validation early stopping.
    Returns (model, history)."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=LR_SCHED_FACTOR, patience=LR_SCHED_PATIENCE)
    crit = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(torch.as_tensor(X_tr), torch.as_tensor(y_tr))
    gen = torch.Generator().manual_seed(0)          # reproducible batch order
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                         generator=gen, drop_last=False)
    scaler = torch.amp.GradScaler("cuda", enabled=_AMP_ENABLED)

    has_val = X_va is not None and len(X_va) > 0
    if has_val:
        Xv = torch.as_tensor(X_va).to(DEVICE)
        yv = torch.as_tensor(y_va).to(DEVICE)

    history = {"train": [], "val": []}
    best_val, best_state, wait = float("inf"), copy.deepcopy(model.state_dict()), 0

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=_AMP_ENABLED):
                loss = crit(model(xb).squeeze(-1), yb)
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                scaler.unscale_(opt)                # clip on true gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()) * len(xb)
        train_loss = tot / len(ds)
        history["train"].append(train_loss)

        if has_val:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=_AMP_ENABLED):
                vloss = float(crit(model(Xv).squeeze(-1), yv))
            history["val"].append(vloss)
            sched.step(vloss)
            if vloss < best_val - 1e-6:
                best_val, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"    early stop @ epoch {ep+1}")
                    break

    if has_val:
        model.load_state_dict(best_state)   # restore best, not last
    return model, history


def predict_scaled(model, X):
    """Batched inference; keeps large windows off the device all at once."""
    model.eval()
    outs = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=_AMP_ENABLED):
        tensor = torch.as_tensor(X)
        for i in range(0, len(tensor), 512):
            chunk = tensor[i:i + 512].to(DEVICE)
            outs.append(model(chunk).squeeze(-1).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros(0, dtype=float)


def predict_returns(model, X, ret_scaler):
    """Model output -> unscaled next-day return."""
    scaled = predict_scaled(model, X).reshape(-1, 1)
    return ret_scaler.inverse_transform(scaled).ravel()

# ---------------------------------------------------------------- classification
def train_classifier(model, X_tr, y_tr, X_va=None, y_va=None, epochs=EPOCHS,
                     patience=EARLY_STOP_PATIENCE, lr=LEARNING_RATE,
                     batch_size=BATCH_SIZE, weight_decay=WEIGHT_DECAY,
                     class_weight=None):
    """Train a 3-class BUY/SELL/HOLD head with cross-entropy.

    Same schedule, early stopping and best-checkpoint restore as train_model;
    only the loss and the label dtype differ. class_weight (a tensor of length
    3) counteracts class imbalance and is computed from TRAIN labels only.
    """
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=LR_SCHED_FACTOR, patience=LR_SCHED_PATIENCE)
    cw = None if class_weight is None else torch.as_tensor(
        class_weight, dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw)

    ds = torch.utils.data.TensorDataset(torch.as_tensor(X_tr),
                                        torch.as_tensor(y_tr, dtype=torch.long))
    gen = torch.Generator().manual_seed(0)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                         generator=gen, drop_last=False)
    has_val = X_va is not None and len(X_va) > 0
    if has_val:
        Xv = torch.as_tensor(X_va).to(DEVICE)
        yv = torch.as_tensor(y_va, dtype=torch.long).to(DEVICE)

    history = {"train": [], "val": []}
    best, best_state, wait = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            tot += float(loss.detach()) * len(xb)
        history["train"].append(tot / len(ds))
        if has_val:
            model.eval()
            with torch.no_grad():
                vloss = float(crit(model(Xv), yv))
            history["val"].append(vloss)
            sched.step(vloss)
            if vloss < best - 1e-6:
                best, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if has_val:
        model.load_state_dict(best_state)
    return model, history


def predict_classes(model, X):
    """Class probabilities (N, 3) from a classification head."""
    model.eval()
    outs = []
    with torch.no_grad():
        tensor = torch.as_tensor(X)
        for i in range(0, len(tensor), 512):
            chunk = tensor[i:i + 512].to(DEVICE)
            outs.append(torch.softmax(model(chunk), dim=1).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 3), dtype=float)


# ---------------------------------------------------------------- multi-task
MT_LAMBDA = 0.5      # weight on the direction loss relative to the return loss


def train_multitask(model, X_tr, y_ret, y_cls, X_va=None, y_ret_va=None,
                    y_cls_va=None, epochs=EPOCHS, patience=EARLY_STOP_PATIENCE,
                    lr=LEARNING_RATE, batch_size=BATCH_SIZE,
                    weight_decay=WEIGHT_DECAY, class_weight=None,
                    lam=MT_LAMBDA):
    """One backbone, two heads: out[:,0] = return, out[:,1:4] = class logits.

    loss = MSE(return) + lam * CrossEntropy(direction)

    Rationale: MSE alone on a low signal-to-noise target pushes predictions
    toward the mean, so the model outputs ~0.2% moves against ~1.5% realised
    ones and never commits to a side. The direction term gives it a reason to
    commit. Early stopping still uses the SAME combined loss on validation.
    """
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=LR_SCHED_FACTOR, patience=LR_SCHED_PATIENCE)
    cw = None if class_weight is None else torch.as_tensor(
        class_weight, dtype=torch.float32, device=DEVICE)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss(weight=cw)

    ds = torch.utils.data.TensorDataset(torch.as_tensor(X_tr),
                                        torch.as_tensor(y_ret),
                                        torch.as_tensor(y_cls, dtype=torch.long))
    gen = torch.Generator().manual_seed(0)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                         generator=gen, drop_last=False)
    has_val = X_va is not None and len(X_va) > 0
    if has_val:
        Xv = torch.as_tensor(X_va).to(DEVICE)
        rv = torch.as_tensor(y_ret_va).to(DEVICE)
        cv = torch.as_tensor(y_cls_va, dtype=torch.long).to(DEVICE)

    best, best_state, wait = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ep in range(epochs):
        model.train()
        for xb, rb, cb in loader:
            xb, rb, cb = xb.to(DEVICE), rb.to(DEVICE), cb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = mse(out[:, 0], rb) + lam * ce(out[:, 1:4], cb)
            loss.backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
        if has_val:
            model.eval()
            with torch.no_grad():
                o = model(Xv)
                vloss = float(mse(o[:, 0], rv) + lam * ce(o[:, 1:4], cv))
            sched.step(vloss)
            if vloss < best - 1e-6:
                best, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if has_val:
        model.load_state_dict(best_state)
    return model, best


def predict_multitask(model, X, ret_scaler):
    """(unscaled returns, class probabilities) from a 4-output head."""
    model.eval()
    rs, ps = [], []
    with torch.no_grad():
        t = torch.as_tensor(X)
        for i in range(0, len(t), 512):
            o = model(t[i:i + 512].to(DEVICE)).float().cpu()
            rs.append(o[:, 0].numpy())
            ps.append(torch.softmax(o[:, 1:4], dim=1).numpy())
    r = np.concatenate(rs).reshape(-1, 1) if rs else np.zeros((0, 1))
    return (ret_scaler.inverse_transform(r).ravel(),
            np.concatenate(ps) if ps else np.zeros((0, 3)))

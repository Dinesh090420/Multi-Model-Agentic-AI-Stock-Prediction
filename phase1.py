"""PHASE 1 - four independently trained BiLSTM experiments.

  E1 YFinance only    OHLCV + technical + price-derived + Elliott Wave
  E2 + Social         E1 + Social_Sentiment + social daily aggregates
  E3 + News           E1 + News_* from the offline FNSPID extract
  E4 Full fusion      E1 + news + social + Combined_Sentiment + Sent_Mom + regime

Each experiment builds its own matrix, sequences, scalers and model; identical
architecture, hyperparameters and seed, so differences trace to features alone.
Purity, weight fingerprints and prediction vectors are asserted, not assumed.

Historical news = data/historical_news.csv, built by:  python phase1.py fetch
Live NewsAPI/StockTwits stay inference-only; nothing is back-dated or invented.
News features are News_* so they never collide with the social Sent_* columns.
"""
from __future__ import annotations

import hashlib, os, sys, time, traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents import DataAgent, SentimentAgent
from config import (BATCH_SIZE, DATA_DIR, DEFAULT_CREDIBILITY, DEVICE,
                    EARLY_STOP_PATIENCE, EPOCHS, EW_FEATURES,
                    HISTORICAL_PRICE_PATH, LEARNING_RATE, PRICE_FEATURES,
                    REGIME_FEATURES, RESULTS_DIR, SENTIMENT_AGG_FEATURES,
                    SEQUENCE_LENGTH, SOURCE_CREDIBILITY, TECH_FEATURES,
                    TICKER_ALIASES, TICKERS)
from evaluation import full_metrics
from features import build_feature_frame, make_dataset, split_bounds
from models import make_model, predict_returns, train_model
from sentiment_analysis import analyze_news, save_cache
from utils import configure_logging, ensure_dirs, get_logger, set_seed

logger = get_logger("phase1")

MODEL_NAME, METRIC_KEYS = "BiLSTM", ["MAE", "RMSE", "MAPE"]
NEWS_PATH = os.getenv("HISTORICAL_NEWS_PATH", os.path.join(DATA_DIR, "historical_news.csv"))
LABEL_PATH = os.getenv("LABELLED_NEWS_PATH", os.path.join(DATA_DIR, "labelled_news_eval.csv"))
PHASE1_DIR = os.path.join(RESULTS_DIR, "phase1")
EXPERIMENT = os.getenv("EXPERIMENT", "all").strip().lower()
_W = 82

NEWS_FEATURES = ["News_Sentiment", "News_Pos_Ratio", "News_Neg_Ratio",
                 "News_Neu_Ratio", "News_Confidence", "News_Volume", "News_Trend"]
_AGG_MAP = {"Avg_Sentiment": "News_Sentiment", "Positive_Ratio": "News_Pos_Ratio",
            "Negative_Ratio": "News_Neg_Ratio", "Neutral_Ratio": "News_Neu_Ratio",
            "Avg_Confidence": "News_Confidence", "Text_Count": "News_Volume",
            "Sentiment_Trend": "News_Trend"}
_COLS = {"date": ["date", "published_at", "datetime", "timestamp", "publish_date"],
         "ticker": ["ticker", "stock", "symbol", "stock name", "stock_symbol"],
         "text": ["headline", "title", "text", "summary", "article", "article_title"],
         "source": ["source", "publisher", "site"], "url": ["url", "link"],
         "label": ["label", "sentiment", "true_label"]}

PRICE_DERIVED = ["Ret_1", "Ret_5", "Vol_10", "Px_SMA20", "BB_PctB", "MACD_Hist", "Vol_Z"]
SOCIAL_COLS = ["Social_Sentiment"] + SENTIMENT_AGG_FEATURES
FUSION_ONLY = ["Combined_Sentiment", "Sent_Mom"] + REGIME_FEATURES
YF_GROUP = PRICE_FEATURES + TECH_FEATURES + PRICE_DERIVED + EW_FEATURES

EXPERIMENTS = [
    ("E1", "YFinance Only", "yfinance_only", YF_GROUP,
     NEWS_FEATURES + SOCIAL_COLS + FUSION_ONLY),
    ("E2", "YFinance + Social", "yfinance_social", YF_GROUP + SOCIAL_COLS,
     NEWS_FEATURES + FUSION_ONLY),
    ("E3", "YFinance + Historical News", "yfinance_news", YF_GROUP + NEWS_FEATURES,
     SOCIAL_COLS + FUSION_ONLY),
    ("E4", "YFinance + News + Social", "full_fusion",
     YF_GROUP + NEWS_FEATURES + SOCIAL_COLS + FUSION_ONLY, []),
]

NO_LABELS = ("Classification metrics unavailable: no independent ground-truth "
             "sentiment labels exist. FinBERT's own output is never its own truth.")


def _find(cols: Dict[str, str], key: str) -> Optional[str]:
    return next((cols[c] for c in _COLS[key] if c in cols), None)


# ---------------------------------------------------------------- historical news
def load_corpus() -> Tuple[Optional[pd.DataFrame], Dict]:
    """Read the offline news CSV. Missing file is reported, never invented."""
    info = {"available": False, "rows": 0, "has_labels": False, "source": "",
            "reason": f"no corpus at '{NEWS_PATH}'; run: python phase1.py fetch"}
    if not os.path.exists(NEWS_PATH):
        logger.warning(info["reason"])
        return None, info
    raw = pd.read_csv(NEWS_PATH)
    cols = {str(c).strip().lower(): c for c in raw.columns}
    d, t, x = (_find(cols, k) for k in ("date", "ticker", "text"))
    if not all((d, t, x)):
        info["reason"] = f"{NEWS_PATH}: need date/ticker/text; found {list(raw.columns)}"
        logger.error(info["reason"])
        return None, info
    src, url, lab = (_find(cols, k) for k in ("source", "url", "label"))
    df = pd.DataFrame({
        "Date": pd.to_datetime(raw[d], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize(),
        "Ticker": raw[t].astype(str).str.strip().str.upper(),
        "Headline": raw[x].astype(str).str.strip(),
        "Source": raw[src].astype(str).str.strip() if src else "unknown",
        "URL": raw[url].astype(str) if url else ""})
    if lab:
        df["True_Label"] = raw[lab].astype(str).str.strip().str.lower()
        info["has_labels"] = True
    df = df.dropna(subset=["Date"])
    df = df[df["Headline"].str.len() >= 15].drop_duplicates(["Ticker", "Date", "Headline"])
    df["Credibility"] = (df["Source"].str.lower().map(SOURCE_CREDIBILITY)
                         .fillna(DEFAULT_CREDIBILITY))
    info.update(available=True, rows=len(df), reason="",
                source=f"OFFLINE CORPUS ({os.path.basename(NEWS_PATH)})")
    logger.info("news corpus: %s articles | %s -> %s | labels: %s", len(df),
                df["Date"].min().date(), df["Date"].max().date(), info["has_labels"])
    return df, info


def news_features(ticker: str, index: pd.DatetimeIndex, corpus) -> Tuple:
    """FinBERT news features aligned to trading days. Gaps stay neutral, never
    forward-filled from a future article."""
    info = {"available": False, "articles": 0, "covered_days": 0,
            "trading_days": len(index), "reason": "no corpus"}
    if corpus is None or corpus.empty:
        return None, None, info
    al = [s.upper() for s in TICKER_ALIASES.get(ticker.upper(), [ticker.upper()])]
    lo, hi = index.min(), index.max()
    sub = corpus[corpus["Ticker"].isin(al) & corpus["Date"].between(lo, hi)]
    if sub.empty:
        info["reason"] = f"no {ticker} articles in {lo.date()} -> {hi.date()}"
        logger.warning("[%s] %s", ticker, info["reason"])
        return None, None, info
    scored, _, agg = analyze_news(sub)                      # existing FinBERT path
    feats = agg.rename(columns=_AGG_MAP)[list(_AGG_MAP.values())].reindex(index).fillna(0.0)
    info.update(available=True, articles=len(sub), reason="",
                covered_days=int(agg.index.isin(index).sum()))
    logger.info("[%s] news: %s articles on %s/%s trading days", ticker,
                info["articles"], info["covered_days"], info["trading_days"])
    return feats, scored, info


_NORM = {"positive": "positive", "pos": "positive", "bullish": "positive",
         "negative": "negative", "neg": "negative", "bearish": "negative",
         "neutral": "neutral", "neu": "neutral"}
_CLS = ["positive", "neutral", "negative"]


def evaluate_finbert(corpus=None) -> Dict:
    """Score FinBERT against INDEPENDENT labels if any exist, else report N/A."""
    out = {"available": False, "reason": NO_LABELS, "source": "", "n": 0}
    texts, truth = [], []
    if os.path.exists(LABEL_PATH):
        raw = pd.read_csv(LABEL_PATH)
        cols = {str(c).strip().lower(): c for c in raw.columns}
        t, l = _find(cols, "text"), _find(cols, "label")
        if t and l:
            texts, truth = raw[t].astype(str).tolist(), raw[l].astype(str).str.lower().tolist()
            out["source"] = f"labelled set ({os.path.basename(LABEL_PATH)})"
    if not texts and corpus is not None and "True_Label" in corpus.columns:
        texts, truth = corpus["Headline"].tolist(), corpus["True_Label"].tolist()
        out["source"] = "labels inside the news corpus"
    if not texts:
        return out
    truth = [_NORM.get(str(t).strip(), "") for t in truth]
    keep = [i for i, t in enumerate(truth) if t in _CLS]
    if not keep:
        out["reason"] = "label column present but no value maps to the 3 classes"
        return out
    texts, truth = [texts[i] for i in keep], [truth[i] for i in keep]
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)
    from sentiment_analysis import FinBERTScorer
    labels, _, _ = FinBERTScorer.instance().score(texts)
    pred = [_NORM.get(str(x).lower(), str(x).lower()) for x in labels]
    kw = dict(labels=_CLS, average="macro", zero_division=0)
    out.update(available=True, n=len(truth), classes=_CLS,
               Accuracy=round(accuracy_score(truth, pred), 4),
               Precision=round(precision_score(truth, pred, **kw), 4),
               Recall=round(recall_score(truth, pred, **kw), 4),
               F1=round(f1_score(truth, pred, **kw), 4),
               confusion=confusion_matrix(truth, pred, labels=_CLS).tolist())
    logger.info("FinBERT on %s labelled texts: acc %.4f macro-F1 %.4f",
                out["n"], out["Accuracy"], out["F1"])
    return out


# ---------------------------------------------------------------- FNSPID fetch
FNSPID_URL = ("https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
              "Stock_news/nasdaq_exteral_data.csv")
FNSPID_API = "https://datasets-server.huggingface.co/filter"
FNSPID_REPO, FNSPID_PAGE = "Zihan1004/FNSPID", 100


def _price_window():
    px = pd.read_csv(HISTORICAL_PRICE_PATH, usecols=lambda c: c.strip().lower() == "date")
    dt = pd.to_datetime(px.iloc[:, 0], errors="coerce", utc=True).dt.tz_localize(None)
    return dt.min().normalize(), dt.max().normalize()


def _rows_to_frame(rows, lo, hi):
    """FNSPID API rows -> our schema, filtered to the price window."""
    df = pd.DataFrame([r["row"] for r in rows])
    if df.empty:
        return df
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], errors="coerce", utc=True)
                  .dt.tz_localize(None).dt.normalize(),
        "ticker": df["Stock_symbol"].astype(str).str.strip().str.upper(),
        "headline": df["Article_title"].astype(str).str.strip(),
        "source": df.get("Publisher", "FNSPID"),
        "url": df.get("Url", "")})
    return out[out["date"].between(lo, hi)]


def _fetch_via_api(keep, lo, hi):
    """Server-side filter - no bulk download. Returns None if unsupported."""
    import json, urllib.parse, urllib.request
    kept = []
    for sym in sorted(keep):
        offset, got = 0, 0
        while True:
            q = urllib.parse.urlencode({
                "dataset": FNSPID_REPO, "config": "default", "split": "train",
                "where": f"\"Stock_symbol\"='{sym}'",
                "limit": FNSPID_PAGE, "offset": offset})
            try:
                with urllib.request.urlopen(f"{FNSPID_API}?{q}", timeout=60) as r:
                    payload = json.load(r)
            except Exception as exc:
                if offset == 0 and not kept:
                    detail = ""
                    if hasattr(exc, "read"):
                        try:
                            detail = exc.read().decode()[:300]
                        except Exception:
                            pass
                    logger.error("datasets-server /filter FAILED (%s: %s) %s",
                                 type(exc).__name__, exc, detail)
                    return None
                logger.warning("[%s] page at offset %s failed: %s", sym, offset, exc)
                break
            rows = payload.get("rows", [])
            if not rows:
                break
            sub = _rows_to_frame(rows, lo, hi)
            if not sub.empty:
                kept.append(sub)
                got += len(sub)
            offset += FNSPID_PAGE
            total = int(payload.get("num_rows_total", 0))
            if offset % 2000 == 0:
                logger.info("[%s] %s/%s rows scanned, %s kept", sym, offset, total, got)
            if offset >= total:
                break
        logger.info("[%s] API: %s articles inside the price window", sym, got)
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()


def _fetch_via_hf(repo, keep, lo, hi):
    """Load a normal-sized HF dataset in full, then filter locally.

    Works for any repo whose rows carry a ticker, a date and headline text -
    column names are resolved flexibly, same as the CSV loader.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets   (needed for HF_DATASET)")
        return None
    logger.info("loading HF dataset %s (full download, then local filter)", repo)
    try:
        df = load_dataset(repo, split="train").to_pandas()
    except Exception as exc:
        logger.error("load_dataset(%s) failed: %s: %s", repo, type(exc).__name__, exc)
        return None
    logger.info("%s: %s rows | columns: %s", repo, f"{len(df):,}", list(df.columns))

    cols = {str(c).strip().lower(): c for c in df.columns}
    d, t, x = (_find(cols, k) for k in ("date", "ticker", "text"))
    if not all((d, t, x)):
        logger.error("%s lacks a date/ticker/text column trio; found %s. It cannot "
                     "drive the news experiment.", repo, list(df.columns))
        return None
    s_c, u_c, l_c = (_find(cols, k) for k in ("source", "url", "label"))

    sym = df[t].astype(str).str.strip().str.upper()
    when = pd.to_datetime(df[d], errors="coerce", utc=True).dt.tz_localize(None)
    logger.info("%s: dates %s -> %s | %s distinct symbols", repo,
                when.min(), when.max(), sym.nunique())
    hit = sym.isin(keep)
    if not hit.any():
        logger.error("none of %s appear in the ticker column; sample values: %s",
                     sorted(keep), list(sym.value_counts().head(10).index))
        return None
    sub = df[hit & when.between(lo, hi)]
    if sub.empty:
        logger.error("%s has the tickers but no rows inside %s -> %s",
                     repo, lo.date(), hi.date())
        return None
    out = pd.DataFrame({
        "date": pd.to_datetime(sub[d], errors="coerce", utc=True)
                  .dt.tz_localize(None).dt.normalize(),
        "ticker": sub[t].astype(str).str.strip().str.upper(),
        "headline": sub[x].astype(str).str.strip(),
        "source": sub[s_c].astype(str) if s_c else repo,
        "url": sub[u_c].astype(str) if u_c else ""})
    if l_c:
        out["label"] = sub[l_c].astype(str).str.strip().str.lower()
        logger.warning("dataset carries a label column - it will enable "
                       "Precision/Recall/F1. Check HOW those labels were made: if "
                       "they came from a model rather than human annotation, the "
                       "resulting scores measure agreement between models, not "
                       "correctness.")
    return out


CANDIDATES = ["KrossKinetic/SP500-Financial-News-Articles-Time-Series",
              "ashraq/financial-news-articles",
              "KalsusEvening/financial-news-headlines",
              "XA7/News-dataset-for-Stock-Market-news-classification"]


def probe(repos=None) -> int:
    """Report whether each candidate dataset can actually drive the news
    experiment: does it carry ticker + date + text, do OUR tickers appear, and
    do its dates cover OUR price window? Prints a verdict per repo."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets")
        return 2
    lo, hi = _price_window()
    keep = {s.upper() for t in TICKERS for s in TICKER_ALIASES.get(t, [t])}
    logger.info("need: %s inside %s -> %s", ", ".join(sorted(keep)), lo.date(), hi.date())

    for repo in (repos or CANDIDATES):
        logger.info("\n%s\n%s\n%s", "=" * 70, repo, "=" * 70)
        try:
            df = load_dataset(repo, split="train").to_pandas()
        except Exception as exc:
            logger.error("  cannot load: %s: %s", type(exc).__name__, exc)
            continue
        cols = {str(c).strip().lower(): c for c in df.columns}
        d, t, x = (_find(cols, k) for k in ("date", "ticker", "text"))
        logger.info("  rows %s | columns %s", f"{len(df):,}", list(df.columns))
        if not all((d, t, x)):
            logger.warning("  VERDICT: unusable - missing %s",
                           [n for n, c in (("date", d), ("ticker", t), ("text", x)) if not c])
            continue
        sym = df[t].astype(str).str.strip().str.upper()
        when = pd.to_datetime(df[d], errors="coerce", utc=True).dt.tz_localize(None)
        logger.info("  dates %s -> %s", when.min(), when.max())
        found = {k: int((sym == k).sum()) for k in sorted(keep)}
        logger.info("  our tickers: %s", found)
        inwin = df[sym.isin(keep) & when.between(lo, hi)]
        if inwin.empty:
            logger.warning("  VERDICT: unusable - 0 rows for our tickers inside the window")
            logger.info("  most common symbols: %s",
                        list(sym.value_counts().head(8).index))
        else:
            per = inwin.groupby(sym[inwin.index]).size().to_dict()
            logger.warning("  VERDICT: USABLE - %s articles in window %s",
                           len(inwin), per)
            logger.info("  fetch it with: HF_DATASET=%s python phase1.py fetch", repo)
    return 0


def _fetch_via_stream(keep, lo, hi):
    """Fallback: stream the 23 GB CSV in chunks, storing only matching rows."""
    src = os.getenv("FNSPID_LOCAL", FNSPID_URL)
    cap = int(os.getenv("FNSPID_MAX_CHUNKS", "0")) or None
    logger.info("falling back to streaming (%s)",
                "local file" if src != FNSPID_URL else "23 GB over HTTP")
    try:
        reader = pd.read_csv(src, chunksize=200_000, on_bad_lines="skip", low_memory=False)
    except Exception as exc:
        logger.error("cannot open FNSPID (%s: %s). Download once and rerun with "
                     "FNSPID_LOCAL=/path/to/nasdaq_exteral_data.csv", type(exc).__name__, exc)
        return None
    kept, scanned, t0 = [], 0, time.time()
    for i, ch in enumerate(reader, 1):
        scanned += len(ch)
        cols = {str(c).strip().lower(): c for c in ch.columns}
        d, t, x = (_find(cols, k) for k in ("date", "ticker", "text"))
        if not all((d, t, x)):
            logger.error("unexpected FNSPID header: %s", list(ch.columns))
            return None
        s_c, u_c = _find(cols, "source"), _find(cols, "url")
        sub = ch[ch[t].astype(str).str.strip().str.upper().isin(keep)]
        if not sub.empty:
            when = pd.to_datetime(sub[d], errors="coerce", utc=True).dt.tz_localize(None)
            sub = sub[when.between(lo, hi)]
            if not sub.empty:
                kept.append(pd.DataFrame({
                    "date": pd.to_datetime(sub[d], errors="coerce", utc=True)
                              .dt.tz_localize(None).dt.normalize(),
                    "ticker": sub[t].astype(str).str.strip().str.upper(),
                    "headline": sub[x].astype(str).str.strip(),
                    "source": sub[s_c].astype(str) if s_c else "FNSPID",
                    "url": sub[u_c].astype(str) if u_c else ""}))
        if i % 20 == 0:
            logger.info("chunk %s | scanned %s | kept %s | %.1f min", i, f"{scanned:,}",
                        sum(map(len, kept)), (time.time() - t0) / 60)
        if cap and i >= cap:
            logger.warning("stopped at %s chunks (FNSPID_MAX_CHUNKS) - PARTIAL extract", i)
            break
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()


def fetch_fnspid() -> int:
    """Build data/historical_news.csv from FNSPID.

    Tries the Hugging Face datasets-server filter endpoint first (server-side
    WHERE, a few thousand rows over HTTP). Falls back to streaming the full CSV
    only if that endpoint is unavailable. FNSPID_STREAM=1 forces the fallback.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    lo, hi = _price_window()
    keep = {s.upper() for t in TICKERS for s in TICKER_ALIASES.get(t, [t])}
    logger.info("window %s -> %s | symbols %s", lo.date(), hi.date(), ", ".join(sorted(keep)))

    hf_repo = os.getenv("HF_DATASET")
    if hf_repo:
        out = _fetch_via_hf(hf_repo, keep, lo, hi)
        if out is None:
            return 2
    elif os.getenv("FNSPID_STREAM"):
        out = _fetch_via_stream(keep, lo, hi)
    else:
        out = _fetch_via_api(keep, lo, hi)
    if out is None and not os.getenv("FNSPID_STREAM"):
        # never drop into a multi-hour download silently - the user opts in
        logger.error("The FNSPID API path did not work, so no news was fetched.")
        logger.error("FNSPID's datasets-server index is broken upstream (an Arrow")
        logger.error("type error on its Russian-language rows), so /filter cannot")
        logger.error("work for that repo. Use a different dataset instead:")
        logger.error("  1. HF_DATASET=<owner/name> python phase1.py fetch   (recommended)")
        logger.error("  2. FNSPID_STREAM=1 python phase1.py fetch   (23 GB, hours)")
        logger.error("  3. download the CSV once, then:")
        logger.error("     FNSPID_STREAM=1 FNSPID_LOCAL=/path/to/nasdaq_exteral_data.csv "
                     "python phase1.py fetch   (~10 min)")
        return 2
    if out is None:
        out = _fetch_via_stream(keep, lo, hi)
    if out is None:
        return 2
    if out.empty:
        logger.error("no matching articles found")
        return 1

    out = (out.drop_duplicates(["ticker", "date", "headline"])
              .sort_values(["ticker", "date"]).reset_index(drop=True))
    out.to_csv(NEWS_PATH, index=False)
    logger.info("kept %s articles | %s -> %s | %s | -> %s", f"{len(out):,}",
                out["date"].min().date(), out["date"].max().date(),
                dict(out["ticker"].value_counts()), NEWS_PATH)
    return 0


# ---------------------------------------------------------------- experiments
def _fingerprint(model) -> str:
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def run_experiment(ticker, exp, name, folder, feature_df, cols, forbidden) -> Dict:
    """Own dataset, own scalers, own freshly built BiLSTM. Nothing carried over."""
    leaked = sorted(set(cols) & set(forbidden))
    if leaked:
        raise AssertionError(f"{exp}: forbidden features present: {leaked}")
    logger.info("\n%s\nPHASE 1 - EXPERIMENT %s   [%s]\n%s -> BiLSTM\n%s",
                "=" * 50, exp[1:], ticker, name, "=" * 50)
    logger.info("Features : %s columns (%s, ...)", len(cols), ", ".join(cols[:4]))
    logger.info("Excluded : %s", ", ".join(forbidden[:5]) + ("..." if len(forbidden) > 5
                else "") if forbidden else "nothing (full fusion)")

    set_seed()
    data = make_dataset(feature_df, feature_columns=cols)
    tr, va, te = split_bounds(data)
    cur, fut, aret = data["current_close"], data["future_close"], data["actual_return"]
    model = make_model(MODEL_NAME, data["input_size"])
    npar = sum(p.numel() for p in model.parameters())
    logger.info("Training...   (%s train sequences, %s parameters)", tr.stop - tr.start, f"{npar:,}")
    logger.info("Validation... (%s sequences, early stopping on val loss)", va.stop - va.start)
    model, hist = train_model(model, data["X"][tr], data["y"][tr], data["X"][va],
                              data["y"][va], epochs=EPOCHS, patience=EARLY_STOP_PATIENCE,
                              lr=LEARNING_RATE, batch_size=BATCH_SIZE)
    pred_ret = predict_returns(model, data["X"][te], data["ret_scaler"])
    pred_close = cur[te] * (1 + pred_ret)
    m = full_metrics(fut[te], pred_close, aret[te], pred_ret)
    logger.info("Testing...    (%s held-out sequences, never seen in training)", len(pred_ret))
    logger.info("\nMAE  : %.4f\nRMSE : %.4f\nMAPE : %.4f %%\nepochs %s | weights %s",
                m["MAE"], m["RMSE"], m["MAPE"], len(hist["train"]), _fingerprint(model))

    d = os.path.join(PHASE1_DIR, folder)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame({"Date": data["dates"][te], "Actual_Close": fut[te],
                  "Predicted_Close": pred_close, "Actual_Return": aret[te],
                  "Predicted_Return": pred_ret}).to_csv(
                      os.path.join(d, f"{ticker}_predictions.csv"), index=False)
    row = {"Stock": ticker, "Exp": exp, "Experiment": name, "n_features": len(cols),
           "n_test": len(pred_ret), "epochs": len(hist["train"]), "params": npar,
           "weights": _fingerprint(model), "pred": pred_ret,
           "constant_features": [c for c in cols if float(feature_df[c].std()) < 1e-12],
           **{k: round(float(m[k]), 4) for k in METRIC_KEYS}}
    pd.DataFrame([{k: row[k] for k in ["Stock", "Experiment"] + METRIC_KEYS}]).to_csv(
        os.path.join(d, f"{ticker}_metrics.csv"), index=False)
    return row


# ---------------------------------------------------------------- sentiment stats
def _stats(scored, daily, labels) -> Dict:
    has = scored is not None and len(scored) and "FinBERT_Label" in scored
    vc = scored["FinBERT_Label"].value_counts() if has else {}
    out = {l: int(vc.get(l, 0)) if has else 0 for l in labels}
    out["Items"] = 0 if scored is None else len(scored)
    out["Avg"] = round(float(scored["FinBERT_Score"].mean()), 4) if has else "N/A"
    out["Cov"] = (f"{float((daily.abs() > 1e-9).mean()) * 100:.1f}%"
                  if daily is not None and len(daily) else "N/A")
    return out


def _merge_sent(acc: Dict, new: Dict) -> Dict:
    """Keep the first NON-EMPTY reading per scope.

    A live scrape can fail for one ticker and succeed for the next (StockTwits
    rate-limits). Reporting only the first ticker would then show zeros for a
    source that actually worked, so each scope is filled independently.
    """
    for k, v in new.items():
        if k not in acc or (acc[k].get("Items", 0) == 0 and v.get("Items", 0) > 0):
            acc[k] = v
    return acc


def sentiment_report(b: Dict) -> Dict:
    daily, live = b.get("sentiment_daily"), (b.get("live") or {})
    ld = live.get("sentiment_daily") if live.get("available") else None
    news_l, soc_l = ["Positive", "Neutral", "Negative"], ["Bullish", "Neutral", "Bearish"]
    rep = {"social_hist": _stats(b.get("social_scored"),
                                 daily["Social_Sentiment"] if daily is not None else None, soc_l)}
    rep["social_hist"]["Src"] = (b.get("social_info") or {}).get("source", "-")
    if live.get("available"):
        for key, scored, col, labs in (("news_live", live.get("news_scored"), "News_Sentiment", news_l),
                                       ("social_live", live.get("social_scored"), "Social_Sentiment", soc_l)):
            rep[key] = _stats(scored, ld[col] if ld is not None else None, labs)
            rep[key]["Src"] = (live.get(key.replace("_live", "_info")) or {}).get("source", "-")
    return rep


# ---------------------------------------------------------------- report
def _rule(ch="-"):
    return ch * _W


def print_report(rows, sent, corpus_info=None, news_status=None, fin_eval=None,
                 news_scored=None) -> pd.DataFrame:
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "pred"} for r in rows])
    L = ["", "#" * _W, "PHASE 1 - BiLSTM FORECASTING RESULTS".center(_W), "#" * _W,
         "", _rule("="), "PER-STOCK RESULTS (each row = one independently trained BiLSTM)",
         _rule("="), f"  {'Stock':<8}{'Experiment':<30}{'MAE':>10}{'RMSE':>10}"
         f"{'MAPE %':>10}{'Feats':>7}{'Weights':>14}", "  " + _rule("-")[:79]]
    for _, sub in df.groupby("Stock", sort=False):
        L += [f"  {r['Stock']:<8}{r['Experiment']:<30}{r['MAE']:>10.4f}{r['RMSE']:>10.4f}"
              f"{r['MAPE']:>10.4f}{r['n_features']:>7}{r['weights']:>14}"
              for _, r in sub.iterrows()] + [""]

    agg = df.groupby("Experiment", sort=False)[METRIC_KEYS].mean().round(4)
    L += [_rule("="), "SUMMARY BY EXPERIMENT (mean of the per-stock values above)",
          _rule("="), f"  {'Experiment':<32}{'MAE':>10}{'RMSE':>10}{'MAPE %':>10}",
          "  " + _rule("-")[:62]]
    L += [f"  {n:<32}{r['MAE']:>10.4f}{r['RMSE']:>10.4f}{r['MAPE']:>10.4f}"
          + ("  <- baseline-compatible" if n == "YFinance Only" else "")
          for n, r in agg.iterrows()]
    L += ["  (per-stock table is primary; averaged MAE/RMSE mix dollar scales.)"]

    dead = sorted({c for r in rows for c in r["constant_features"]})
    if dead:
        L += ["", "  WARNING - zero-variance columns: " + ", ".join(dead),
              "  A constant column carries no information; any metric change from",
              "  adding only such columns is initialisation noise, not signal."]

    ci, fe = corpus_info or {}, fin_eval or {}
    L += ["", _rule("="), "HISTORICAL NEWS SOURCE (offline, training)", _rule("=")]
    if ci.get("available"):
        L += [f"  Source : {ci['source']}  ({ci['rows']:,} articles)",
              "  Live NewsAPI was NOT used for training and nothing was back-dated."]
        L += [f"    {t:<7} {s['articles']:>6} articles | {s['covered_days']}/"
              f"{s['trading_days']} trading days covered | "
              f"{s['trading_days'] - s['covered_days']} days without news"
              if s.get("available") else f"    {t:<7} UNAVAILABLE - {s.get('reason','')}"
              for t, s in (news_status or {}).items()]
    else:
        L += [f"  UNAVAILABLE - {ci.get('reason', 'no corpus configured')}",
              "  E3 and the news half of E4 are not a genuine news test until then."]

    L += ["", _rule("="), "FINBERT CLASSIFICATION EVALUATION", _rule("=")]
    if fe.get("available"):
        L += [f"  Ground truth : {fe['source']} ({fe['n']} labelled texts)",
              f"  Accuracy {fe['Accuracy']} | Precision {fe['Precision']} | "
              f"Recall {fe['Recall']} | F1 {fe['F1']}  (macro, 3-class)",
              "  Confusion (rows=true, cols=pred; " + ", ".join(fe["classes"]) + "):"]
        L += [f"    {c:<9} " + "  ".join(f"{v:>5}" for v in row)
              for c, row in zip(fe["classes"], fe["confusion"])]
    else:
        L += ["  Accuracy / Precision / Recall / F1 : N/A", f"  {fe.get('reason', NO_LABELS)}",
              "  Optional: data/labelled_news_eval.csv (columns text,label)."]

    def _scope_row(scored, stats, head):
        """(counts, detail line) for one scope; zeros + N/A when absent."""
        if scored is not None and len(scored) and "FinBERT_Label" in scored:
            vc = scored["FinBERT_Label"].value_counts()
            return ([int(vc.get(h, 0)) for h in head],
                    f"    items {len(scored)} | avg sentiment "
                    f"{scored['FinBERT_Score'].mean():.4f} | source: offline corpus")
        if stats:
            return ([stats[h] for h in head],
                    f"    items {stats['Items']} | avg sentiment {stats['Avg']} | "
                    f"coverage {stats['Cov']} | source: {stats['Src']}")
        return [0, 0, 0], "    items 0 | avg sentiment N/A | coverage 0.0% | source: unavailable"

    for title, head, scopes in (
        ("NEWS SENTIMENT", ("Positive", "Neutral", "Negative"),
         [("Historical corpus (training)", news_scored, None),
          ("Live NewsAPI (LIVE-ONLY)", None, sent.get("news_live"))]),
        ("SOCIAL SENTIMENT", ("Bullish", "Neutral", "Bearish"),
         [("Historical (stock_tweets.csv)", None, sent.get("social_hist")),
          ("Live StockTwits (LIVE-ONLY)", None, sent.get("social_live"))])):
        L += ["", _rule("="), title, _rule("="), f"  {NO_LABELS}", "",
              f"  {'Scope':<30}{head[0]:>9}{head[1]:>9}{head[2]:>9}"
              f"{'Prec':>7}{'Rec':>7}{'F1':>7}", "  " + _rule("-")[:78]]
        for label, scored, stats in scopes:
            vals, detail = _scope_row(scored, stats, head)
            L += [f"  {label:<30}{vals[0]:>9}{vals[1]:>9}{vals[2]:>9}"
                  f"{'N/A':>7}{'N/A':>7}{'N/A':>7}", detail]

    L += ["", _rule("="), "VALIDATION - SEPARATE TRAINING CONFIRMATION", _rule("=")]
    for exp, name, _f, _c, _x in EXPERIMENTS:
        sub = df[df["Exp"] == exp]
        L.append(f"  [ ] {name:<32} NOT RUN (EXPERIMENT={EXPERIMENT})" if sub.empty else
                 f"  [\u2713] {name:<32} trained separately on {len(sub)} stock(s); "
                 f"weights: {', '.join(sub['weights'])}")

    best = agg["MAPE"].idxmin()
    L += ["", _rule("="), "BEST EXPERIMENT", _rule("="),
          f"  Lowest mean MAPE: {best} ({agg.loc[best, 'MAPE']:.4f} %)",
          "", _rule("="), "BASELINE-PAPER COMPARISON POINT", _rule("="),
          "  Use 'YFinance Only' (E1) - price features only. Compare on MAE /",
          "  RMSE / MAPE only; sentiment statistics are not comparable.",
          "  'YFinance + News + Social' (E4) is the Phase-1 hybrid result.",
          "", "#" * _W, ""]

    logger.info("\n".join(L))
    os.makedirs(PHASE1_DIR, exist_ok=True)
    with open(os.path.join(PHASE1_DIR, "phase1_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    df.drop(columns=["constant_features"]).to_csv(
        os.path.join(PHASE1_DIR, "phase1_summary.csv"), index=False)
    return df


# ---------------------------------------------------------------- entry point
def _selected():
    if EXPERIMENT in ("all", ""):
        return EXPERIMENTS
    picked = [e for e in EXPERIMENTS if e[2] == EXPERIMENT]
    if not picked:
        raise SystemExit(f"unknown EXPERIMENT={EXPERIMENT!r}; use one of: "
                         + ", ".join(e[2] for e in EXPERIMENTS) + ", all")
    return picked


def main() -> int:
    configure_logging()
    ensure_dirs()
    os.makedirs(PHASE1_DIR, exist_ok=True)
    set_seed()
    logger.info("PHASE 1 | %s only | device %s | %s | seq %s | chronological splits",
                MODEL_NAME, DEVICE, ", ".join(TICKERS), SEQUENCE_LENGTH)

    corpus, corpus_info = load_corpus()
    fin_eval = evaluate_finbert(corpus)
    news_status, news_scored, rows, sent = {}, None, [], {}

    for ticker in TICKERS:
        logger.info("\n%s\nSTOCK: %s\n%s", "=" * _W, ticker, "=" * _W)
        try:
            b = SentimentAgent().run(DataAgent().run(ticker))   # both unchanged
            fdf = build_feature_frame(b["stock_df"], b["sentiment_daily"], b["regime_df"])
            nf, nsc, ninfo = news_features(ticker, fdf.index, corpus)
            news_status[ticker] = ninfo
            for c in NEWS_FEATURES:
                fdf[c] = nf[c].to_numpy() if nf is not None else 0.0
            news_scored = news_scored if news_scored is not None else nsc
            sent = _merge_sent(sent, sentiment_report(b))

            local = [run_experiment(ticker, *e[:3], fdf, *e[3:]) for e in _selected()]
            fps = [r["weights"] for r in local]
            if len(set(fps)) != len(fps):
                raise AssertionError(f"[{ticker}] duplicate model weights: {fps}")
            preds = [r.pop("pred") for r in local]
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    if np.allclose(preds[i], preds[j]):
                        raise AssertionError(f"[{ticker}] experiments {i+1}/{j+1} "
                                             f"produced identical predictions")
            logger.info("\n[%s] VERIFIED: %s separate BiLSTM models, %s distinct "
                        "fingerprints, %s distinct prediction vectors.",
                        ticker, len(local), len(local), len(local))
            rows += local
        except Exception as exc:
            logger.error("[%s] skipped: %s: %s", ticker, type(exc).__name__, exc)
            logger.debug("traceback:\n%s", traceback.format_exc())

    save_cache()
    if not rows:
        logger.error("no experiments completed - aborting")
        return 1
    print_report(rows, sent, corpus_info, news_status, fin_eval, news_scored)
    logger.info("Phase-1 artefacts under %s", PHASE1_DIR)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        configure_logging()
        sys.exit(fetch_fnspid())
    if cmd == "probe":
        configure_logging()
        sys.exit(probe(sys.argv[2:] or None))
    sys.exit(main())

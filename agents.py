"""Five-agent architecture. Each agent is thin and delegates to the modules.

Data regimes are strictly separated:
  TRAINING  reads offline CSVs only (no API call can influence a fitted model)
  LIVE      Yahoo Finance + NewsAPI + StockTwits, used only for inference

DataAgent       -> load historical CSVs (+ optional live snapshot), technical +
                   Elliott-wave + regime features
PredictionAgent -> fuse features, validation HPO, train LSTM/GRU/Transformer + baseline,
                   directional-accuracy-weighted ensemble, validation-optimized
                   thresholds, walk-forward, metrics
DecisionAgent   -> BUY/SELL/HOLD (optimized thresholds) + calibrated confidence
ReportAgent     -> explainable per-ticker report

Cross-sectional ranking across all tickers is done in main.py (ranking.py) after
every ticker's DecisionAgent has run.
"""
import numpy as np
import pandas as pd

from config import (EPOCHS, EARLY_STOP_PATIENCE, DO_HPO, HPO_GRID, HPO_EPOCHS,
                    SEED, PRICE_FEATURES, TECH_FEATURES, EW_FEATURES,
                    SENTIMENT_FEATURES, SENTIMENT_AGG_FEATURES, REGIME_FEATURES,
                    DERIVED_FEATURES, LIVE_PREDICTION_ENABLED, MAX_TWEETS_PER_DAY,
                    SEQUENCE_LENGTH)
from data_collection import (HistoricalDataStore, fetch_live_prices,
                             load_historical_prices, load_historical_tweets)
from news_social import (fetch_live_news, fetch_live_social,
                         social_engagement_metrics)
from utils import get_logger

logger = get_logger("agents")
from technical_indicators import add_technical_indicators, add_elliott_wave_features
from market_regime import detect_regime
from sentiment_analysis import (analyze_news, analyze_social, build_daily_sentiment)
from technical_indicators import add_technical_indicators as _add_tech
from features import build_feature_frame, make_dataset, split_bounds, FEATURE_COLUMNS
from models import (make_model, train_model, predict_returns, DEFAULT_HP)
from evaluation import full_metrics, walk_forward, to_signal, optimize_thresholds

MODEL_NAMES = ["LSTM", "GRU", "Transformer"]


class DataAgent:
    """Historical CSVs for training; an optional live snapshot for inference."""

    @staticmethod
    def _engineer(stock_df):
        """Technical + Elliott-wave + regime features on an OHLCV frame."""
        stock_df = add_technical_indicators(stock_df)
        stock_df = add_elliott_wave_features(stock_df)
        return stock_df, detect_regime(stock_df)

    @staticmethod
    def _cap_per_day(tweets, ticker):
        """Deterministic per-day cap: keeps every day represented, bounds cost."""
        if not MAX_TWEETS_PER_DAY or tweets.empty:
            return tweets
        capped = (tweets.sort_values("Date")
                        .groupby(tweets["Date"].dt.normalize(), group_keys=False)
                        .head(MAX_TWEETS_PER_DAY))
        if len(capped) < len(tweets):
            logger.info("[%s] tweets capped at %s/day: %s -> %s rows",
                        ticker, MAX_TWEETS_PER_DAY, len(tweets), len(capped))
        return capped.reset_index(drop=True)

    def run(self, ticker):
        logger.info("[%s] DataAgent: loading historical datasets", ticker)
        stock_df, src_info = load_historical_prices(ticker)
        stock_df, regime_df = self._engineer(stock_df)

        tweets_df, social_info = load_historical_tweets(ticker)
        tweets_df = self._cap_per_day(tweets_df, ticker)
        social_info["posts"] = int(len(tweets_df))
        if not tweets_df.empty:
            HistoricalDataStore.instance().merge_audit(
                ticker, stock_df.index, tweets_df["Date"])

        # Historical regime carries no news channel: the CSV corpus is social
        # text only. News_Sentiment stays neutral and Combined renormalizes onto
        # the social channel (see sentiment_analysis.build_daily_sentiment).
        news_df = pd.DataFrame(columns=["Date", "Source", "Credibility", "Headline"])
        news_info = {"ticker": ticker, "mode": "historical",
                     "source": "n/a (historical corpus is social text only)",
                     "is_real": False, "reason": "", "articles": 0}

        bundle = {"ticker": ticker, "mode": "historical",
                  "stock_df": stock_df, "regime_df": regime_df,
                  "news_df": news_df, "social_df": tweets_df,
                  "src_info": src_info, "news_info": news_info,
                  "social_info": social_info,
                  "merge_audit": HistoricalDataStore.instance().audit.get(ticker, {})}

        if LIVE_PREDICTION_ENABLED:
            bundle["live"] = self._collect_live(ticker)
        return bundle

    def _collect_live(self, ticker):
        """Live snapshot for inference. Failures degrade to None, never raise."""
        logger.info("[%s] DataAgent: collecting live snapshot", ticker)
        live = {"available": False, "reason": ""}
        prices, price_info = fetch_live_prices(ticker)
        if prices.empty:
            live["reason"] = price_info.get("reason", "no live prices")
            return live
        stock_df, regime_df = self._engineer(prices)
        news_df, news_info = fetch_live_news(ticker)
        social_df, social_info = fetch_live_social(ticker)
        live.update(available=True, stock_df=stock_df, regime_df=regime_df,
                    news_df=news_df, social_df=social_df,
                    price_info=price_info, news_info=news_info,
                    social_info=social_info)
        return live


class SentimentAgent:
    """FinBERT scoring + daily aggregation for the historical corpus and the
    live snapshot. Identical code path for both, so live and training features
    are computed by exactly the same logic."""

    @staticmethod
    def _score(news_df, social_df, social_info, index):
        news_scored, news_daily, news_agg = analyze_news(news_df)
        social_scored, social_daily, social_agg = analyze_social(social_df, social_info)
        daily = build_daily_sentiment(news_daily, social_daily, index,
                                      social_agg=social_agg, news_agg=news_agg)
        return news_scored, social_scored, daily

    def run(self, b):
        t = b["ticker"]
        logger.info("[%s] SentimentAgent: FinBERT scoring (historical)", t)
        news_scored, social_scored, sentiment_daily = self._score(
            b["news_df"], b["social_df"], b["social_info"], b["stock_df"].index)
        b.update(news_scored=news_scored, social_scored=social_scored,
                 sentiment_daily=sentiment_daily)

        live = b.get("live")
        if live and live.get("available"):
            logger.info("[%s] SentimentAgent: FinBERT scoring (live)", t)
            ln, ls, ldaily = self._score(live["news_df"], live["social_df"],
                                         live["social_info"], live["stock_df"].index)
            live.update(news_scored=ln, social_scored=ls, sentiment_daily=ldaily,
                        engagement=social_engagement_metrics(ls))
        return b


class PredictionAgent:
    def run(self, b):
        t = b["ticker"]
        print(f"  [PredictionAgent] {t}: fusing features + training...")
        feature_df = build_feature_frame(b["stock_df"], b["sentiment_daily"], b["regime_df"])
        data = make_dataset(feature_df)
        tr, va, te = split_bounds(data)
        ret_scaler = data["ret_scaler"]

        cur, fut, aret = data["current_close"], data["future_close"], data["actual_return"]
        histories, chosen_hp, trained = {}, {}, {}
        val_r, test_r = {}, {}

        for m in MODEL_NAMES:
            hp = self._select_hp(m, data, tr, va, ret_scaler) if DO_HPO else DEFAULT_HP
            chosen_hp[m] = hp
            model = make_model(m, data["input_size"], hp)
            model, hist = train_model(model, data["X"][tr], data["y"][tr],
                                      data["X"][va], data["y"][va],
                                      epochs=EPOCHS, patience=EARLY_STOP_PATIENCE,
                                      lr=hp["lr"], batch_size=hp["batch_size"])
            histories[m] = hist
            trained[m] = model                      # kept for feature-importance probes
            val_r[m] = predict_returns(model, data["X"][va], ret_scaler)
            test_r[m] = predict_returns(model, data["X"][te], ret_scaler)

        # ensemble weights from VALIDATION directional accuracy (down-weight <=chance)
        val_dir = {m: float(np.mean((aret[va] > 0) == (val_r[m] > 0))) for m in MODEL_NAMES}
        raw = {m: max(val_dir[m] - 0.5, 0.01) for m in MODEL_NAMES}
        s = sum(raw.values())
        weights = {m: raw[m] / s for m in MODEL_NAMES}

        ens_w_test = sum(weights[m] * test_r[m] for m in MODEL_NAMES)
        ens_w_val = sum(weights[m] * val_r[m] for m in MODEL_NAMES)
        eq_test_ret = np.mean([test_r[m] for m in MODEL_NAMES], axis=0)
        eq_val_ret = np.mean([val_r[m] for m in MODEL_NAMES], axis=0)

        # ---- validation-based selection over the FULL candidate set ----
        # Candidates: each individual model + both ensemble variants. The final
        # "Ensemble" predictor is the candidate with the best VALIDATION
        # directional accuracy (tie-broken toward combinations, which have lower
        # variance, then by validation RMSE). This guarantees the reported
        # Ensemble is at least as good as the best individual model ON
        # VALIDATION - the only leakage-free place such a guarantee can exist;
        # any method claiming it on the TEST set would be selecting on test.
        cand_val = {**val_r, "Weighted": ens_w_val, "Equal": eq_val_ret}
        cand_test = {**test_r, "Weighted": ens_w_test, "Equal": eq_test_ret}

        def _val_score(name):
            v = cand_val[name]
            dacc = float(np.mean((aret[va] > 0) == (v > 0)))
            rmse = float(np.sqrt(np.mean((cur[va] * (1 + v) - fut[va]) ** 2)))
            return (dacc, name in ("Weighted", "Equal"), -rmse)

        chosen = max(cand_val, key=_val_score)
        ens_val_ret, ens_test_ret = cand_val[chosen], cand_test[chosen]
        val_dir_ens = float(np.mean((aret[va] > 0) == (ens_val_ret > 0)))
        if chosen == "Weighted":
            ens_variant = "dir-acc-weighted combination (selected on validation)"
        elif chosen == "Equal":
            weights = {m: 1.0 / len(MODEL_NAMES) for m in MODEL_NAMES}
            ens_variant = "equal-weight combination (selected on validation)"
        else:
            weights = {m: 1.0 if m == chosen else 0.0 for m in MODEL_NAMES}
            ens_variant = f"single best model: {chosen} (selected on validation)"

        # optimize BUY/SELL thresholds on VALIDATION ensemble returns (no test data)
        buy_thr, sell_thr = optimize_thresholds(ens_val_ret, aret[va])

        # baseline (naive momentum): tomorrow's return = today's return.
        # Price and return are CONSISTENT: base_close = cur * (1 + base_ret),
        # so RMSE and directional accuracy describe the SAME baseline model.
        base_ret = (cur[te] - data["prev_close"][te]) / data["prev_close"][te]
        base_close = cur[te] * (1 + base_ret)

        pred_returns = {**{m: test_r[m] for m in MODEL_NAMES},
                        "Ensemble": ens_test_ret, "Ensemble_Equal": eq_test_ret,
                        "Baseline": base_ret}
        preds = {**{m: cur[te] * (1 + test_r[m]) for m in MODEL_NAMES},
                 "Ensemble": cur[te] * (1 + ens_test_ret),
                 "Ensemble_Equal": cur[te] * (1 + eq_test_ret),
                 "Baseline": base_close}

        actual_close, actual_ret = fut[te], aret[te]
        metrics = {k: full_metrics(actual_close, preds[k], actual_ret,
                                   pred_returns[k], buy_thr, sell_thr) for k in preds}

        print(f"  [PredictionAgent] {t}: walk-forward validation...")
        wf = walk_forward(data, MODEL_NAMES, ret_scaler, chosen_hp)

        print(f"  [PredictionAgent] {t}: feature-group importance (validation)...")
        feat_importance, const_groups = self._group_importance(
            trained, weights, data, va, ret_scaler, aret, val_dir_ens)

        # honest data-coverage diagnostics: what fraction of each split actually
        # had nonzero sentiment? (Real NewsAPI/StockTwits only cover recent
        # weeks, so early splits are expected to be ~0% - report it, don't hide it.)
        cs = b["sentiment_daily"]["Combined_Sentiment"]
        sent_coverage = {}
        for nm, sl in (("train", tr), ("val", va), ("test", te)):
            d = pd.DatetimeIndex(data["dates"][sl])
            seg = cs.reindex(d).fillna(0.0)
            sent_coverage[nm] = float((seg.abs() > 1e-9).mean())

        if b.get("live") and b["live"].get("available"):
            b["live"]["prediction"] = self._live_forecast(b, data, trained, weights)

        b.update(feature_df=feature_df, data=data, slices=(tr, va, te),
                 histories=histories, chosen_hp=chosen_hp, weights=weights,
                 ens_variant=ens_variant, feat_importance=feat_importance,
                 const_groups=const_groups, sent_coverage=sent_coverage,
                 val_dir=val_dir, val_dir_ensemble=val_dir_ens, wf=wf,
                 buy_thr=float(buy_thr), sell_thr=float(sell_thr),
                 test_dates=data["dates"][te], actual_close=actual_close,
                 actual_return=actual_ret, current_close=cur[te],
                 preds=preds, pred_returns=pred_returns, metrics=metrics)
        return b

    @staticmethod
    def _group_importance(trained, weights, data, va, ret_scaler, aret, base_acc):
        """Grouped permutation importance on VALIDATION data (leakage-free).

        For each feature group, the group's columns are shuffled ACROSS validation
        windows (breaking their relationship with the target while preserving the
        marginal distribution) and the drop in ensemble directional accuracy is
        recorded. importance > 0  => the group carries predictive signal;
        importance ~ 0 => the models learned little from it. Uses the already-
        trained models; the test set is never touched.
        """
        groups = {"Price": PRICE_FEATURES, "Technical": TECH_FEATURES,
                  "ElliottWave": EW_FEATURES, "Sentiment": SENTIMENT_FEATURES,
                  "Regime": REGIME_FEATURES, "Derived": DERIVED_FEATURES}
        rng = np.random.default_rng(SEED)
        Xva = data["X"][va]
        imp, constant = {}, set()
        for gname, cols in groups.items():
            idx = [FEATURE_COLUMNS.index(c) for c in cols]
            # a group with ZERO VARIANCE in the validation inputs cannot be
            # measured: permuting identical values changes nothing, so its
            # importance is 0 BY CONSTRUCTION, not because it was tested and
            # found useless. Flag it so the report can say so honestly.
            if float(Xva[:, :, idx].std()) < 1e-12:
                imp[gname] = 0.0
                constant.add(gname)
                continue
            Xp = Xva.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, :, idx] = Xp[perm][:, :, idx]
            ens = sum(weights[m] * predict_returns(trained[m], Xp, ret_scaler)
                      for m in trained)
            acc = float(np.mean((aret[va] > 0) == (ens > 0)))
            imp[gname] = round(base_acc - acc, 4)
        return dict(sorted(imp.items(), key=lambda kv: -kv[1])), constant

    @staticmethod
    def _live_forecast(b, data, trained, weights):
        """Predict the next-day return from the LIVE window using trained models.

        The live features are transformed with the TRAINING scaler (never
        refitted - refitting on live data would leak and invalidate the model).
        Because the training corpus and the live window can be years apart, the
        share of scaled values outside the training range is reported as an
        explicit distribution-shift diagnostic rather than hidden.
        """
        t = b["ticker"]
        live = b["live"]
        try:
            fdf = build_feature_frame(live["stock_df"], live["sentiment_daily"],
                                      live["regime_df"])
            if len(fdf) < SEQUENCE_LENGTH:
                return {"available": False,
                        "reason": f"only {len(fdf)} live rows, need {SEQUENCE_LENGTH}"}
            window = fdf[FEATURE_COLUMNS].to_numpy(dtype=float)[-SEQUENCE_LENGTH:]
            scaled = data["feat_scaler"].transform(window)
            oor = float(np.mean((scaled < -0.05) | (scaled > 1.05)))
            X = scaled.reshape(1, SEQUENCE_LENGTH, -1).astype(np.float32)

            per_model = {m: float(predict_returns(trained[m], X,
                                                  data["ret_scaler"])[0])
                         for m in trained}
            ens_ret = float(sum(weights[m] * per_model[m] for m in per_model))
            cur_close = float(fdf["Close"].iloc[-1])
            return {"available": True, "reason": "",
                    "as_of": str(pd.Timestamp(fdf.index[-1]).date()),
                    "current_close": cur_close,
                    "predicted_return": ens_ret,
                    "predicted_price": cur_close * (1.0 + ens_ret),
                    "per_model": per_model,
                    "out_of_range_frac": oor,
                    "sentiment": float(live["sentiment_daily"]
                                       ["Combined_Sentiment"].iloc[-1]),
                    "regime": str(live["regime_df"]["Regime_Label"].iloc[-1]),
                    "trend_regime": str(live["regime_df"]["Trend_Regime"].iloc[-1])}
        except Exception as exc:
            logger.error("[%s] live forecast failed: %s: %s", t,
                         type(exc).__name__, exc)
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _select_hp(name, data, tr, va, ret_scaler):
        """Compact validation search: pick hp maximizing (val dir-acc, -val rmse)."""
        cur, fut, aret = data["current_close"], data["future_close"], data["actual_return"]
        best = None
        for hp in HPO_GRID:
            m = make_model(name, data["input_size"], hp)
            m, _ = train_model(m, data["X"][tr], data["y"][tr],
                               data["X"][va], data["y"][va],
                               epochs=HPO_EPOCHS, patience=HPO_EPOCHS,
                               lr=hp["lr"], batch_size=hp["batch_size"])
            vr = predict_returns(m, data["X"][va], ret_scaler)
            diracc = float(np.mean((aret[va] > 0) == (vr > 0)))
            rmse = float(np.sqrt(np.mean((cur[va] * (1 + vr) - fut[va]) ** 2)))
            score = (diracc, -rmse)
            if best is None or score > best[0]:
                best = (score, hp)
        return best[1]


class DecisionAgent:
    def run(self, b):
        t = b["ticker"]
        print(f"  [DecisionAgent] {t}: recommendation + confidence...")
        pr = b["pred_returns"]
        buy_thr, sell_thr = b["buy_thr"], b["sell_thr"]
        ens_ret = float(pr["Ensemble"][-1])
        model_last = np.array([pr[m][-1] for m in MODEL_NAMES])

        # confidence: grounded in validation reliability + agreement + signal strength
        agreement = 1.0 - min(model_last.std() / 0.02, 1.0)
        # signal strength is measured against the threshold on the SAME side
        # as the prediction; with per-side (asymmetric) thresholds, scaling a
        # SELL-strength prediction by the BUY threshold would mis-state it.
        side_thr = buy_thr if ens_ret >= 0 else abs(sell_thr)
        magnitude = min(abs(ens_ret) / max(side_thr, 0.003), 1.0)
        val_perf = b["val_dir_ensemble"]
        confidence = float(np.clip(0.4 * val_perf + 0.35 * agreement + 0.25 * magnitude, 0, 1))

        sent = float(b["sentiment_daily"]["Combined_Sentiment"].iloc[-1])
        trend = b["regime_df"]["Trend_Regime"].iloc[-1]
        regime_label = b["regime_df"]["Regime_Label"].iloc[-1]

        # base signal from optimized thresholds, then SYMMETRIC sentiment/regime overrides
        rec = to_signal(ens_ret, buy_thr, sell_thr)
        note = ""
        if rec == "BUY" and (sent < -0.5 or trend == "Bear"):
            rec, note = "HOLD", "upside conflicts with bearish sentiment/regime"
        elif rec == "SELL" and (sent > 0.5 or trend == "Bull"):
            rec, note = "HOLD", "downside conflicts with bullish sentiment/regime"
        elif rec == "HOLD" and ens_ret > 0 and sent > 0.3 and trend == "Bull":
            rec, note = "BUY", "mild upside confirmed by bullish sentiment + regime"
        elif rec == "HOLD" and ens_ret < 0 and sent < -0.3 and trend == "Bear":
            rec, note = "SELL", "mild downside confirmed by bearish sentiment + regime"

        # risk = TRAILING 20-day realized vol, excluding the final day whose
        # return is the very outcome being predicted (keeps the decision layer
        # strictly past-only rather than using the whole realized test window).
        past = np.asarray(b["actual_return"])[:-1]
        risk = float(np.std(past[-20:])) if len(past) else 0.0
        cur_price = float(b["current_close"][-1])
        pred_price = float(b["preds"]["Ensemble"][-1])
        actual_price = float(b["actual_close"][-1])

        explanation = (
            f"Ensemble predicts {ens_ret*100:+.2f}% next-day return "
            f"(validation-tuned BUY>={buy_thr*100:+.2f}% / SELL<={sell_thr*100:+.2f}%). "
            f"Model agreement {agreement:.2f}, validation dir-acc {val_perf:.2f}. "
            f"Combined sentiment {sent:+.2f}; regime {regime_label}."
            + (f" Adjusted: {note}." if note else "")
        )

        live = b.get("live") or {}
        lp = live.get("prediction") or {}
        if lp.get("available"):
            b["live_decision"] = self._live_decision(b, lp, buy_thr, sell_thr,
                                                     val_perf)

        b["decision"] = {
            "recommendation": rec, "confidence": round(confidence, 3),
            "current_price": round(cur_price, 2),
            "predicted_price": round(pred_price, 2), "actual_price": round(actual_price, 2),
            "predicted_return": ens_ret, "combined_sentiment": sent,
            "risk": risk, "regime": regime_label, "trend_regime": trend,
            "agreement": round(float(agreement), 3), "val_dir": round(val_perf, 3),
            "buy_thr": buy_thr, "sell_thr": sell_thr, "explanation": explanation,
        }
        return b


def _live_decision_impl(self, b, lp, buy_thr, sell_thr, val_perf):
    """Same decision rules as the historical path, applied to live features."""
    ens_ret = float(lp["predicted_return"])
    vals = np.array(list(lp["per_model"].values()), dtype=float)
    agreement = 1.0 - min(float(vals.std()) / 0.02, 1.0)
    side_thr = buy_thr if ens_ret >= 0 else abs(sell_thr)
    magnitude = min(abs(ens_ret) / max(side_thr, 0.003), 1.0)
    confidence = float(np.clip(0.4 * val_perf + 0.35 * agreement + 0.25 * magnitude,
                               0, 1))
    # a large distribution shift between training and live inputs makes the
    # forecast less trustworthy; discount confidence rather than hide it
    shift_penalty = float(np.clip(lp.get("out_of_range_frac", 0.0), 0.0, 1.0))
    confidence *= (1.0 - 0.5 * shift_penalty)

    sent, trend = float(lp["sentiment"]), lp["trend_regime"]
    rec = to_signal(ens_ret, buy_thr, sell_thr)
    note = ""
    if rec == "BUY" and (sent < -0.5 or trend == "Bear"):
        rec, note = "HOLD", "upside conflicts with bearish sentiment/regime"
    elif rec == "SELL" and (sent > 0.5 or trend == "Bull"):
        rec, note = "HOLD", "downside conflicts with bullish sentiment/regime"
    elif rec == "HOLD" and ens_ret > 0 and sent > 0.3 and trend == "Bull":
        rec, note = "BUY", "mild upside confirmed by bullish sentiment + regime"
    elif rec == "HOLD" and ens_ret < 0 and sent < -0.3 and trend == "Bear":
        rec, note = "SELL", "mild downside confirmed by bearish sentiment + regime"

    return {"recommendation": rec, "confidence": round(confidence, 3),
            "as_of": lp["as_of"], "current_price": round(lp["current_close"], 2),
            "predicted_price": round(lp["predicted_price"], 2),
            "predicted_return": ens_ret, "combined_sentiment": sent,
            "regime": lp["regime"], "agreement": round(float(agreement), 3),
            "distribution_shift": round(shift_penalty, 3), "note": note}


DecisionAgent._live_decision = _live_decision_impl


class ReportAgent:
    def run(self, b):
        d = b["decision"]
        t = b["ticker"]
        news_scored, social_scored = b["news_scored"], b["social_scored"]
        news_counts = news_scored["FinBERT_Label"].value_counts().to_dict()
        soc_counts = social_scored["FinBERT_Label"].value_counts().to_dict()
        si = b["social_info"]
        soc_method = ("FinBERT per post" if si.get("text_available")
                      else "provider aggregate (no raw text)")

        L = []
        L.append("=" * 66)
        L.append(f"  {t}  -  EXPLAINABLE REPORT")
        L.append("=" * 66)
        L.append("  DATA")
        L.append(f"    Source          : {b['src_info']['source']}  "
                 f"({'REAL' if b['src_info']['is_real'] else 'SYNTHETIC'})")
        if b['src_info'].get('reason'):
            L.append(f"    Fallback reason : {b['src_info']['reason']}")
        L.append(f"    OHLCV records   : {b['src_info']['records']}   "
                 f"range {b['src_info']['date_range']}")
        L.append(f"    Regime          : TRAINING = offline CSV only "
                 f"(no API influences any fitted model)")
        ma = b.get("merge_audit") or {}
        if ma:
            L.append(f"    Merge audit     : {ma.get('matched_days', 0)}/"
                     f"{ma.get('trading_days', 0)} trading days have text; "
                     f"{ma.get('tweet_days_without_session', 0)} text days on "
                     f"non-trading dates; "
                     f"{ma.get('trading_days_without_tweets', 0)} days without text")
        L.append("  NEWS")
        L.append(f"    Source          : {b['news_info']['source']}  "
                 f"({'REAL' if b['news_info']['is_real'] else 'SYNTHETIC'})   "
                 f"articles {b['news_info'].get('articles', 0)}")
        L.append(f"    Pos/Neu/Neg     : {news_counts.get('Positive',0)}/"
                 f"{news_counts.get('Neutral',0)}/{news_counts.get('Negative',0)}")
        L.append("  SOCIAL")
        L.append(f"    Source          : {si['source']}  "
                 f"({'REAL' if si['is_real'] else 'SYNTHETIC'})   posts {si.get('posts',0)}")
        if si.get('reason'):
            L.append(f"    Fallback reason : {si['reason']}")
        L.append(f"    Sentiment method: {soc_method}")
        L.append(f"    Engagement      : {si.get('engagement_desc','N/A')}")
        L.append(f"    Bull/Neu/Bear   : {soc_counts.get('Bullish',0)}/"
                 f"{soc_counts.get('Neutral',0)}/{soc_counts.get('Bearish',0)}")
        L.append("  SENTIMENT")
        L.append(f"    News={b['sentiment_daily']['News_Sentiment'].iloc[-1]:+.3f}  "
                 f"Social={b['sentiment_daily']['Social_Sentiment'].iloc[-1]:+.3f}  "
                 f"Combined={d['combined_sentiment']:+.3f}")
        cov = b["sent_coverage"]
        L.append(f"    Coverage (days with nonzero sentiment): "
                 f"train {cov['train']*100:.1f}%  val {cov['val']*100:.1f}%  "
                 f"test {cov['test']*100:.1f}%")
        L.append(f"  MARKET REGIME     : {d['regime']}")
        L.append(f"  TEST PERIOD       : {pd.Timestamp(b['test_dates'][0]).date()} -> "
                 f"{pd.Timestamp(b['test_dates'][-1]).date()}  "
                 f"({len(b['test_dates'])} days)")
        L.append("  MODEL RESULTS (test)")
        for m in ["Baseline", "LSTM", "GRU", "Transformer", "Ensemble"]:
            L.append(f"    {m:12s}: dir-acc {b['metrics'][m]['Directional_Accuracy']:.3f}  "
                     f"RMSE {b['metrics'][m]['RMSE']:.3f}  "
                     f"MAPE {b['metrics'][m]['MAPE']:.2f}%  "
                     f"Sharpe {b['metrics'][m]['Sharpe']:+.2f}")
        L.append(f"    Ensemble type   : {b['ens_variant']}")
        L.append(f"    Ensemble weights: " +
                 ", ".join(f"{m} {b['weights'][m]:.2f}" for m in MODEL_NAMES))
        wf = b["wf"]
        wf_ens = wf.get("Ensemble", [])
        if wf_ens:
            L.append(f"    Walk-forward    : ensemble dir-acc "
                     f"{np.mean(wf_ens):.3f} +/- {np.std(wf_ens):.3f} "
                     f"over {wf['folds']} chronological folds")
        L.append("  FEATURE-GROUP CONTRIBUTION (validation permutation importance)")
        L.append("    (drop in ensemble dir-acc when the group is shuffled; ~0 =>")
        L.append("     the models learned little from that group)")
        for g, v in b["feat_importance"].items():
            note = ("  [constant in validation window - no variance to measure;"
                    " see Coverage above]" if g in b["const_groups"] else "")
            L.append(f"    {g:12s}: {v:+.4f}{note}")
        L.append("  FINAL DECISION")
        L.append(f"    Current ${d['current_price']:.2f} -> Predicted ${d['predicted_price']:.2f}  "
                 f"({d['predicted_return']*100:+.2f}% of current)   "
                 f"Realized ${d['actual_price']:.2f}")
        L.append(f"    Thresholds      : BUY>={d['buy_thr']*100:+.2f}%  "
                 f"SELL<={d['sell_thr']*100:+.2f}%  (validation-optimized)")
        L.append(f"    Recommendation  : {d['recommendation']}   confidence {d['confidence']*100:.1f}%")
        L.append(f"    Reason          : {d['explanation']}")

        ld = b.get("live_decision")
        live = b.get("live") or {}
        if ld:
            eng = live.get("engagement") or {}
            L.append("  LIVE SIGNAL (Yahoo Finance + NewsAPI + StockTwits)")
            L.append(f"    As of           : {ld['as_of']}")
            L.append(f"    Current ${ld['current_price']:.2f} -> Predicted "
                     f"${ld['predicted_price']:.2f}  "
                     f"({ld['predicted_return']*100:+.2f}%)")
            L.append(f"    Recommendation  : {ld['recommendation']}   "
                     f"confidence {ld['confidence']*100:.1f}%"
                     + (f"   ({ld['note']})" if ld["note"] else ""))
            L.append(f"    Live sentiment  : {ld['combined_sentiment']:+.3f}   "
                     f"regime {ld['regime']}")
            if eng:
                L.append(f"    StockTwits      : {eng['Message_Count']:.0f} msgs  "
                         f"bullish {eng['Bullish_Pct']:.0f}% / bearish "
                         f"{eng['Bearish_Pct']:.0f}%  likes {eng['Total_Likes']:.0f} "
                         f"(avg {eng['Avg_Likes']:.1f})  replies "
                         f"{eng['Total_Replies']:.0f} (avg {eng['Avg_Replies']:.1f})")
            L.append(f"    Distribution shift: {ld['distribution_shift']*100:.1f}% of "
                     f"live feature values fall outside the training range")
            if ld["distribution_shift"] > 0.25:
                L.append("      WARNING: large shift - the live window differs "
                         "materially from the training era; treat with caution.")
        elif live.get("reason"):
            L.append(f"  LIVE SIGNAL       : unavailable ({live['reason']})")
        L.append("=" * 66)
        text = "\n".join(L)
        print(text)
        b["report_text"] = text
        return b
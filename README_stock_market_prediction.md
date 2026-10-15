# 📈 Stock Market Prediction with Deep Learning and Sentiment Analysis

A research project that experiments with stock-return forecasting and **BUY / SELL / HOLD** classification by combining historical market data, technical features, news sentiment, and social-media sentiment.

## 🔍 Project Overview

The project evaluates three deep-learning models—**BiLSTM, GRU, and Transformer**—and combines their forecasts through an ensemble. It includes two experiment stages:

- **Phase 1:** compares feature configurations, including price-only inputs and combinations with news and social sentiment.
- **Phase 2:** compares BiLSTM, GRU, Transformer, and ensemble forecasts, then evaluates BUY / SELL / HOLD decisions.

The current experiments cover **AAPL, MSFT, and GOOGL**. This is an academic/research project, not a trading service or financial advice.

## ✨ Key Features

- Historical stock-price and volume data processing
- Technical-feature engineering and market-regime features
- FinBERT-based sentiment scoring for financial text
- Historical tweet and financial-news feature integration
- BiLSTM, GRU, and Transformer forecasting models
- Ensemble forecasts and BUY / SELL / HOLD classification
- Evaluation reports, CSV outputs, and presentation graphs

Live-data integrations are optional: Yahoo Finance price retrieval is attempted by the code, while live NewsAPI data requires a `NEWSAPI_KEY`. The StockTwits integration requires Playwright. The historical experiments can use the local CSV datasets described below.

## 🧠 Workflow

1. Load and validate historical price and tweet datasets.
2. Prepare technical, sentiment, and market-regime features.
3. Run Phase 1 feature/ablation experiments.
4. Train and compare BiLSTM, GRU, and Transformer models in Phase 2.
5. Combine forecasts and evaluate BUY / SELL / HOLD classifications.
6. Generate summary reports and graphs.

## 🧰 Tech Stack

- Python
- PyTorch
- pandas and NumPy
- scikit-learn
- Hugging Face Transformers / FinBERT
- yfinance
- Matplotlib

## 📁 Project Structure

```text
stock-market-prediction-final/
├── main.py                 # Phase 1 experiments
├── phase2.py               # Forecasting and decision experiments
├── summary.py              # Prints a summary from saved results
├── result_plots.py         # Generates presentation graphs
├── data_collection.py      # Loads and prepares historical datasets
├── sentiment_analysis.py   # FinBERT sentiment processing
├── technical_indicators.py # Technical indicators
├── models.py               # Model definitions
├── agents.py               # Data, sentiment, prediction and decision workflow
├── features.py             # Feature preparation
├── market_regime.py        # Market-regime features
├── evaluation.py           # Evaluation utilities
├── requirements.txt        # Python dependencies
├── data/                   # Local datasets; not included in GitHub
└── results/                # Generated reports, CSVs and graphs
```

## 📊 Datasets

Place these files in a local `data/` directory before running the experiments:

| File | Purpose |
|---|---|
| `stock_yfinance_data.csv` | Historical OHLCV stock data |
| `stock_tweets.csv` | Historical tweets/social text |
| `historical_news.csv` | Historical financial news |

`data/all-data.csv` is only needed if you run the optional `all_data.py` script to create `labelled_news_eval.csv`.

**The CSV datasets are not included in this repository.** Obtain them from a source you are permitted to use, and do not upload private or licensed datasets without permission. Keep the local `data/` folder excluded through `.gitignore`.

## ⚙️ Setup

Run these commands from the project root. On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Confirm that the three required CSVs are in `data/` before starting the experiments.

## ▶️ Run the Project

Run each command separately and wait for it to finish:

```bash
python main.py
python phase2.py
RESULTS=results python summary.py
python result_plots.py
```

The optional news-label preparation step is only needed if `data/all-data.csv` is available and you specifically want the evaluation dataset:

```bash
python all_data.py
```

## 📦 Outputs

The scripts write their results under `results/`, including:

- `results/phase1/` — experiment predictions, metric CSVs, summary and report
- `results/phase2/` — model comparisons, forecasts, classification metrics, confusion matrices and report
- `results/phase1/fig1_phase1_ablation.png` — Phase 1 comparison graph
- `results/phase2/fig2_phase2_forecasting.png` — forecasting comparison graph
- `results/phase2/fig3_decision_quality.png` — decision-quality graph
- `results/phase2/fig4_error_structure.png` — error-structure graph
- `results/logs/` — run logs

Some graphs are skipped when their required input CSV is not present.

## 📈 Current Experimental Results

The local run completed for AAPL, MSFT, and GOOGL. The ensemble's test-set forecasting results were:

| Stock | Ensemble MAE | Ensemble RMSE | Ensemble MAPE | Price-only baseline MAPE |
|---|---:|---:|---:|---:|
| AAPL | 2.8065 | 3.5654 | 1.8118% | 1.7016% |
| MSFT | 3.6594 | 4.8877 | 1.4345% | 1.2309% |
| GOOGL | 1.8109 | 2.4257 | 1.7032% | 1.5471% |

In this run, the price-only baseline had lower MAE, RMSE, and MAPE than the ensemble for all three stocks. BUY / SELL / HOLD classification also varied by stock and requires further improvement. These are experimental test results, not evidence of profitability. The historical test data in this run ends on **2022-09-29**, so the output should not be treated as a current market prediction.

## ⚠️ Notes and Limitations

- Performance depends on the quality, date range, and format of the supplied datasets.
- Missing a NewsAPI key disables the live-news source; it does not provide historical news automatically.
- StockTwits collection requires Playwright and its browser installation.
- The first FinBERT run downloads the public pretrained model and requires internet access.
- Forecast errors are reported in the scale used by the implementation; review the prediction target and units before comparing models.
- This repository is for educational and research use only. It does not provide financial advice or guarantee future performance.

## 🚀 Future Improvements

- Improve BUY / SELL / HOLD class balance and decision calibration.
- Compare models against simple baselines on the same time splits.
- Extend the historical period and validate across additional market regimes.
- Pin dependency versions and document a fully reproducible environment.

## 📄 License

See the `LICENSE` file. Replace any remaining placeholder text in that file before publishing.

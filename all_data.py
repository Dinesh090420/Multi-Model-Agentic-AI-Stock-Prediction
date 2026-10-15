"""One-time conversion: PhraseBank 'all-data.csv' -> data/labelled_news_eval.csv

Input : data/all-data.csv   (Financial PhraseBank, no header, latin-1; columns label, text)
Output: data/labelled_news_eval.csv  (columns text, label)

Run from the project root:  python all_data.py
"""
import os
import sys

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SRC = os.path.join(DATA_DIR, "all-data.csv")
OUT = os.path.join(DATA_DIR, "labelled_news_eval.csv")


def main() -> int:
    if not os.path.exists(SRC):
        print(f"ERROR: {SRC} not found. Put all-data.csv inside the data/ folder.")
        return 1
    df = pd.read_csv(SRC, header=None, names=["label", "text"], encoding="latin-1")
    df[["text", "label"]].to_csv(OUT, index=False)
    print(len(df), df["label"].value_counts().to_dict())
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

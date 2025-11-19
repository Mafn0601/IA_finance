# backend/train_models_from_csv.py
import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

BASE_PATH = Path(r"C:\Users\Marco\Documents\dados")
MODELS_DIR = Path(__file__).resolve().parent / "modelos"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = MODELS_DIR / "summary.json"

ATIVOS = ["WINZ25", "WDOZ25", "PETR4"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 150

def ensure_datetime(df):
    df["datetime"] = pd.to_datetime(df["time"])
    return df

def add_indicators(df):
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    df["ema8"] = close.ewm(span=8).mean()
    df["ema21"] = close.ewm(span=21).mean()
    df["ema50"] = close.ewm(span=50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    df["ret"] = close.pct_change()
    df["vol20"] = df["ret"].rolling(20).std()

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    df = df.dropna().reset_index(drop=True)
    return df

def make_labels(df, threshold=0.0006):
    df = df.copy()
    df["next_close"] = df["close"].shift(-1)
    df["next_ret"] = (df["next_close"] - df["close"]) / df["close"]

    def label(x):
        if x > threshold:
            return 1
        if x < -threshold:
            return -1
        return 0

    df["label"] = df["next_ret"].apply(label)
    df = df.dropna().reset_index(drop=True)
    return df

def train_model_for_asset(symbol):
    print(f"\n=== TREINO: {symbol} ===")
    p_csv = BASE_PATH / f"{symbol}_5min.csv"

    if not p_csv.exists():
        print(f"Arquivo de candles não encontrado: {p_csv}")
        return None

    df = pd.read_csv(p_csv)
    df = ensure_datetime(df)

    df = add_indicators(df)
    df = make_labels(df)

    if len(df) < 200:
        print("Poucos dados para treino:", len(df))
        return None

    feat_cols = [
        "ema8","ema21","ema50",
        "rsi14","macd","macd_signal",
        "bb_width","vol20","stoch_k","stoch_d"
    ]

    X = df[feat_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    model_path = MODELS_DIR / f"{symbol}.pkl"
    joblib.dump(model, model_path)

    print(f"Modelo salvo: {model_path}")

    return {
        "symbol": symbol,
        "acc_test": float(acc),
        "n_samples": len(df),
        "trained_at": datetime.now().isoformat()
    }

def main():
    summary = []
    for ativo in ATIVOS:
        info = train_model_for_asset(ativo)
        if info:
            summary.append(info)

    if summary:
        with open(SUMMARY_PATH, "w", encoding="utf8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nSumário salvo em: {SUMMARY_PATH}")

if __name__ == "__main__":
    main()

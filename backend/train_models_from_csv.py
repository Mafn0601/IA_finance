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

# ativos a treinar (modifique se quiser)
ATIVOS = ["WINZ25", "WDOX25", "PETR4"]

# parametros de treino
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 150

#--------------------------------------------------
# helpers de indicadores / features
#--------------------------------------------------
def ensure_datetime(df):
    # aceita coluna "time" unix epoch em segundos, ou "datetime"
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    if "time" in df.columns:
        try:
            # se time for inteiro unix seconds
            df["datetime"] = pd.to_datetime(df["time"].astype(int), unit="s")
            return df
        except Exception:
            pass
    # fallback: tentar parse em colunas date+time
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
        return df
    raise ValueError("Não achei coluna de tempo reconhecível (esperado: time unix ou datetime)")

def add_indicators(df):
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df.get("tick_volume", df.get("volume", pd.Series(np.ones(len(df))))) 

    df["ema8"] = close.ewm(span=8, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    # Volatility (std of returns)
    df["ret"] = close.pct_change()
    df["vol20"] = df["ret"].rolling(20).std()

    # Stochastic
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    df = df.dropna().reset_index(drop=True)
    return df

def make_labels(df, threshold=0.0006):
    """
    Gera label baseado no movimento do próximo candle:
      +1 compra se next_return > threshold
      -1 venda if next_return < -threshold
      0 neutro otherwise
    threshold default 0.0006 (0.06%) - ajuste conforme ativo
    """
    df = df.copy()
    df["next_close"] = df["close"].shift(-1)
    df["next_ret"] = (df["next_close"] - df["close"]) / (df["close"] + 1e-9)
    def f(x):
        if x > threshold:
            return 1
        if x < -threshold:
            return -1
        return 0
    df["label"] = df["next_ret"].apply(f)
    df = df.dropna().reset_index(drop=True)
    return df

def features_and_target(df):
    # escolhe colunas de feature
    feat_cols = [
        "ema8","ema21","ema50",
        "rsi14","macd","macd_signal",
        "bb_width","vol20","stoch_k","stoch_d"
    ]
    X = df[feat_cols].astype(float).values
    y = df["label"].astype(int).values
    return X, y

#--------------------------------------------------
# rotina de treino por ativo
#--------------------------------------------------
def train_model_for_asset(symbol):
    print(f"\n=== TREINO: {symbol} ===")
    # tenta csv ou xlsx
    p_csv = BASE_PATH / f"{symbol}.csv"
    p_xlsx = BASE_PATH / f"{symbol}.xlsx"
    if p_csv.exists():
        df = pd.read_csv(p_csv, engine="python")
    elif p_xlsx.exists():
        df = pd.read_excel(p_xlsx)
    else:
        print(f"Arquivo não encontrado para {symbol}: {p_csv} ou {p_xlsx}")
        return None

    try:
        df = ensure_datetime(df)
    except Exception as e:
        print("Erro parse tempo:", e)
        return None

    # garantir colunas necessárias
    required = ["open","high","low","close"]
    for r in required:
        if r not in df.columns:
            print(f"Coluna faltando {r} em {symbol}")
            return None

    df = add_indicators(df)
    df = make_labels(df)

    # balanceamento simples: remover abundância de 0s? (opcional)
    # keep all for now

    X, y = features_and_target(df)
    if len(y) < 100:
        print("Poucos exemplos depois de preparar features:", len(y))
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    # metrics
    ypred = model.predict(X_test)
    acc = accuracy_score(y_test, ypred)
    print(f"{symbol} acc test: {acc:.4f}")
    print(classification_report(y_test, ypred))

    # salvar modelo
    model_path = MODELS_DIR / f"{symbol}.pkl"
    joblib.dump(model, model_path)
    print(f"Modelo salvo: {model_path}")

    # salvar summary parcial
    summary = {
        "symbol": symbol,
        "acc_test": float(acc),
        "n_samples": int(len(y)),
        "model_path": str(model_path),
        "trained_at": datetime.now().isoformat()
    }
    return summary

def main():
    all_summary = []
    for s in ATIVOS:
        try:
            info = train_model_for_asset(s)
            if info:
                all_summary.append(info)
        except Exception as e:
            print(f"Erro treinando {s}: {e}")
    if all_summary:
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(all_summary, f, indent=2, ensure_ascii=False)
        print(f"Sumário salvo em {SUMMARY_PATH}")
    else:
        print("Nenhum modelo treinado.")

if __name__ == "__main__":
    main()

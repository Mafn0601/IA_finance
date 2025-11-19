# backend/convert_ticks_to_5min.py
import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Marco\Documents\dados")

ATIVOS = {
    "WINZ25": "WINZ25.csv",
    "WDOZ25": "WDOZ25.csv",
    "PETR4": "PETR4.csv"
}

def converter_para_candles_5min(arquivo_csv, nome_ativo):
    print(f"\n=== Convertendo {nome_ativo} para candles de 5min ===")

    df = pd.read_csv(arquivo_csv)

    colunas = {"time", "open", "high", "low", "close", "real_volume"}
    if not colunas.issubset(df.columns):
        raise ValueError(f"Arquivo {arquivo_csv} sem colunas necessárias.")

    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    candles = df.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "real_volume": "sum"
    }).dropna()

    out_path = BASE_DIR / f"{nome_ativo}_5min.csv"
    candles.to_csv(out_path)

    print(f"Arquivo gerado: {out_path}")
    print(f"Total de candles: {len(candles)}")

    return out_path


if __name__ == "__main__":
    print("=== Conversão de Ticks → 5 min ===")

    for ativo, arquivo in ATIVOS.items():
        caminho = BASE_DIR / arquivo
        if caminho.exists():
            converter_para_candles_5min(caminho, ativo)
        else:
            print(f"[AVISO] Arquivo não encontrado: {caminho}")

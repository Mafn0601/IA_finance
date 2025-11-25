# baixar_dados_reais_dukascopy.py  ← RODA 1x NA VIDA (10 minutos)
import pandas as pd
import requests
import os
from datetime import datetime
import time

BASE = r"C:\Users\Marco\Documents\dados_reais"
os.makedirs(BASE, exist_ok=True)

# WIN = EURUSD como proxy (melhor que ^BVSP)
# WDO = EURUSD (1:1 com dólar)
# PETR4 = PETR4.SA (tem na Dukascopy)

ativos = {
    "WINZ25": "EURUSD",
    "WDOZ25": "EURUSDCHF", 
    "PETR4":  "PETR4"
}

def baixar_m5(symbol, ano):
    url = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{ano}/M5.csv"
    try:
        df = pd.read_csv(url, header=None, names=['datetime','open','high','low','close','volume'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except:
        return pd.DataFrame()

for nome, symbol in ativos.items():
    print(f"\nBaixando {nome} ({symbol}) M5 real 2018–2024...")
    todos = []
    for ano in range(2018, 2025):
        print(f"  {ano}...", end=" ")
        df_ano = baixar_m5(symbol, ano)
        if not df_ano.empty:
            todos.append(df_ano)
            print("OK")
        else:
            print("sem dados")
        time.sleep(0.5)
    
    if todos:
        final = pd.concat(todos, ignore_index=True)
        final = final[['datetime','open','high','low','close','volume']]
        caminho = f"{BASE}\\{nome}_M5_REAL_2018-2024.csv"
        final.to_csv(caminho, index=False)
        print(f"{nome} → {len(final):,} candles REAIS salvos!")
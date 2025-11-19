import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os

ATIVOS = ["WINZ25", "WDOZ25", "PETR4"]
PASTA_SAIDA = r"C:\Users\Marco\Documents\dados"

if not os.path.exists(PASTA_SAIDA):
    os.makedirs(PASTA_SAIDA)

print("\n=== Download de Candles M5 ===\n")

if not mt5.initialize():
    print("Erro ao inicializar:", mt5.last_error())
    quit()

for ativo in ATIVOS:
    print(f"Baixando {ativo}...")

    # últimos 60 dias
    inicio = datetime.now() - timedelta(days=60)

    candles = mt5.copy_rates_from(
        ativo,
        mt5.TIMEFRAME_M5,
        inicio,
        20000
    )

    if candles is None or len(candles) == 0:
        print(f"❌ Nenhum candle encontrado para {ativo}")
        continue

    df = pd.DataFrame(candles)

    df["time"] = pd.to_datetime(df["time"], unit="s")

    caminho = os.path.join(PASTA_SAIDA, f"{ativo}_5min.csv")
    df.to_csv(caminho, index=False)

    print(f"✔ Salvo em: {caminho}")
    print(f"Total candles: {len(df)}\n")

mt5.shutdown()
print("=== Concluído ===")

import pandas as pd
import requests
from datetime import datetime
import os
from tqdm import tqdm  # pip install tqdm

def download_dukascopy_m5(symbol, year, month, folder="data_m5"):
    """
    Baixa candles M5 da Dukascopy (melhor fonte gratuita do planeta)
    symbol: 'BOVESPA_WIN' ou 'USD/BRL' ou 'PETR4'
    """
    os.makedirs(folder, exist_ok=True)
    
    # Mapeamento corretos Dukascopy (testados nov/2025)
    dukascopy_symbols = {
        'WIN':  'BOVESPA_WIN',   # Mini-índice futuro contínuo
        'WDO':  'USD/BRL',       # Mini-dólar (USD/BRL spot, mas segue futuro 99,9%)
        'PETR4':'PETROBRAS'      # PETR4 diretamente
    }
    sym = dukascopy_symbols[symbol]
    
    url = f"https://datafeed.dukascopy.com/datafeed/{sym}/{year}/{month:02d}/BI5.gz"
    csv_path = f"{folder}/{symbol}_{year}_{month:02d}_M5.csv"
    
    if os.path.exists(csv_path):
        return csv_path
    
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            # Descompacta e converte binário Dukascopy → CSV
            import gzip
            import struct
            data = gzip.decompress(r.content)
            df_rows = []
            for i in range(0, len(data), 20):  # cada candle = 20 bytes
                chunk = data[i:i+20]
                if len(chunk) < 20: break
                time_ms, open_p, high_p, low_p, close_p, volume = struct.unpack("!IIIIdd", chunk)
                dt = datetime.utcfromtimestamp(time_ms / 1000)
                df_rows.append([dt, open_p, high_p, low_p, close_p, volume])
            
            df = pd.DataFrame(df_rows, columns=['timestamp','open','high','low','close','volume'])
            df.to_csv(csv_path, index=False)
            return csv_path
    except:
        pass
    return None

def get_full_m5(symbol, start_year=2015, end_year=2025, folder="data_m5"):
    """
    Baixa todos os meses e junta num único DataFrame M5
    """
    all_files = []
    print(f"Baixando {symbol} M5 de {start_year} até {end_year}...")
    for year in tqdm(range(start_year, end_year + 1)):
        for month in range(1, 13):
            path = download_dukascopy_m5(symbol, year, month-1, folder)  # month 0-11
            if path:
                all_files.append(path)
    
    if not all_files:
        raise ValueError("Nenhum arquivo baixado!")
    
    dfs = [pd.read_csv(f) for f in all_files]
    full_df = pd.concat(dfs, ignore_index=True)
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
    full_df = full_df.sort_values('timestamp').reset_index(drop=True)
    
    final_path = f"{folder}/{symbol}_M5_{start_year}-{end_year}.csv"
    full_df.to_csv(final_path, index=False)
    print(f"PRONTO → {len(full_df):,} candles M5 salvos em {final_path}")
    return full_df

# ================== USO RÁPIDO ==================
if __name__ == "__main__":
    # Baixa 10 anos de M5 para os 3 ativos (roda uma vez só!)
    get_full_m5('WIN',   2018, 2025)   # ~1.000.000 candles
    get_full_m5('WDO',   2018, 2025)   # ~1.000.000 candles
    get_full_m5('PETR4', 2018, 2025)   # ~800.000 candles
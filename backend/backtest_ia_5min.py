# backtest_rotativo_melhor_unico_json.py
"""
Backtest rotativo – mantém apenas UMA melhor estratégia por ativo no Postgres
- Roda continuamente, rotacionando entre ATIVOS
- Testa combinações aleatórias por ativo
- Substitui a estratégia salva apenas se novo setup for melhor (lucro primary, dd secondary, pf tertiary)
- Gera gráfico equity do top1 por ativo
- Paper-trading simulado (candle-a-candle) usando o melhor setup encontrado na rodada
- Salva parâmetros, equity e trades_details completos em JSON no banco
"""

import os
import time
import json
import warnings
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import psycopg2

warnings.filterwarnings("ignore")

# ========== CONFIGURAÇÃO ==========
BASE_PATH = r"C:\Users\Marco\Documents\dados"
RESULTADOS_PATH = os.path.join(BASE_PATH, "backtest_resultados")
os.makedirs(RESULTADOS_PATH, exist_ok=True)

ATIVOS = ["WINZ25", "WDOX25", "PETR4"]
CAPITAL_INICIAL = 1000.0
DD_MAX = 20.0

COMBINACOES_POR_ATIVO = 60
COMBINACOES_EXTRA_WINZ = 40

PAUSA_SEGUNDOS = 600  # 10 minutos

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "ia_finance",
    "user": "postgres",
    "password": "admin12345"
}

HIST_CANDLES = 2000
PLOT_PATH = RESULTADOS_PATH

# ========== UTILITÁRIOS ==========
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def connect_db():
    return psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )

def criar_tabela_db():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS resultados_backtest (
        id SERIAL PRIMARY KEY,
        ativo TEXT UNIQUE,
        modo TEXT,
        lucro NUMERIC,
        drawdown NUMERIC,
        winrate NUMERIC,
        profit_factor NUMERIC,
        rr NUMERIC,
        params JSONB,
        equity JSONB,
        trades_details JSONB,
        data TIMESTAMP DEFAULT NOW()
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS paper_state (
        ativo TEXT PRIMARY KEY,
        saldo NUMERIC,
        ultimo_trade JSONB,
        updated_at TIMESTAMP
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

def get_saved_setup(ativo):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT id, ativo, modo, lucro::float, drawdown::float, profit_factor::float, params FROM resultados_backtest WHERE ativo=%s", (ativo,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def save_or_replace_best(ativo, modo, metricas, params, equity, trades_details):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT id, lucro::float, drawdown::float, profit_factor::float FROM resultados_backtest WHERE ativo=%s", (ativo,))
    row = cur.fetchone()
    if row is None:
        cur.execute("""
            INSERT INTO resultados_backtest (ativo, modo, lucro, drawdown, winrate, profit_factor, rr, params, equity, trades_details, data)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (ativo, modo, metricas['lucro'], metricas['dd'], metricas['winrate'], metricas['profit_factor'], metricas['rr'], json.dumps(params), json.dumps(equity), json.dumps(trades_details), datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
        log(f"{ativo}: nenhum setup salvo antes — novo setup salvo (lucro {metricas['lucro']}, dd {metricas['dd']})")
        return True
    else:
        saved_id, saved_lucro, saved_dd, saved_pf = row[0], float(row[1]), float(row[2]), float(row[3])
        replace = False
        if metricas['lucro'] > saved_lucro:
            replace = True
        elif metricas['lucro'] == saved_lucro and metricas['dd'] < saved_dd:
            replace = True
        elif metricas['lucro'] == saved_lucro and metricas['dd'] == saved_dd and metricas['profit_factor'] > saved_pf:
            replace = True

        if replace:
            cur.execute("DELETE FROM resultados_backtest WHERE id=%s", (saved_id,))
            cur.execute("""
                INSERT INTO resultados_backtest (ativo, modo, lucro, drawdown, winrate, profit_factor, rr, params, equity, trades_details, data)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (ativo, modo, metricas['lucro'], metricas['dd'], metricas['winrate'], metricas['profit_factor'], metricas['rr'], json.dumps(params), json.dumps(equity), json.dumps(trades_details), datetime.now()))
            conn.commit()
            cur.close()
            conn.close()
            log(f"{ativo}: setup substituído! antigo(lucro {saved_lucro}, dd {saved_dd}) -> novo(lucro {metricas['lucro']}, dd {metricas['dd']})")
            return True
        else:
            cur.close()
            conn.close()
            log(f"{ativo}: novo setup não melhor que o salvo (salvo: lucro {saved_lucro}, dd {saved_dd} | novo: lucro {metricas['lucro']}, dd {metricas['dd']})")
            return False

def save_paper_state(ativo, saldo, ultimo_trade):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO paper_state (ativo, saldo, ultimo_trade, updated_at)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (ativo) DO UPDATE SET saldo = EXCLUDED.saldo, ultimo_trade = EXCLUDED.ultimo_trade, updated_at = EXCLUDED.updated_at
    """, (ativo, saldo, json.dumps(ultimo_trade), datetime.now()))
    conn.commit()
    cur.close()
    conn.close()

# ======================
# Funções de dados e indicadores
# ======================
def carregar_csv(ativo):
    path = os.path.join(BASE_PATH, f"{ativo}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, engine="python")
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), format="%Y.%m.%d %H:%M:%S", errors='coerce')
        else:
            df = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, engine="python")
            df.columns = ["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"]
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format="%Y.%m.%d %H:%M:%S", errors='coerce')
        for c in ['open','high','low','close']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['volume'] = pd.to_numeric(df['tickvol'] if 'tickvol' in df.columns else (df['volume'] if 'volume' in df.columns else 1), errors='coerce')
        df = df.dropna(subset=['datetime','open','high','low','close']).sort_values('datetime').reset_index(drop=True)
        return df[['datetime','open','high','low','close','volume']]
    except Exception as e:
        log(f"carregar_csv erro {ativo}: {e}")
        return None

def carregar_mt5(ativo, candles=HIST_CANDLES):
    try:
        if not mt5.initialize():
            if not mt5.initialize():
                log("MT5 não inicializou")
                return None
        rates = mt5.copy_rates_from_pos(ativo, mt5.TIMEFRAME_M5, 0, candles)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df['volume'] = df.get('volume', df.get('tick_volume', 1))
        return df[['datetime','open','high','low','close','volume']]
    except Exception as e:
        log(f"carregar_mt5 erro {ativo}: {e}")
        return None

def carregar_candles(ativo, prefer_mt5=True):
    df = carregar_mt5(ativo) if prefer_mt5 else None
    if df is None or df.empty:
        df = carregar_csv(ativo)
    return df

def calcular_indicadores(df):
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume']
    df['ema8'] = close.ewm(span=8, adjust=False).mean()
    df['ema21'] = close.ewm(span=21, adjust=False).mean()
    df['ema50'] = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = -delta.where(delta<0,0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100/(1+rs))
    df['macd'] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    tr = pd.concat([(high-low).abs(), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    dm_pos = (high - high.shift()).clip(lower=0)
    dm_neg = (low.shift() - low).clip(lower=0)
    df['plus_di'] = 100*dm_pos/df['atr']
    df['minus_di'] = 100*dm_neg/df['atr']
    return df

def sinal_confidence_row(row, params):
    buy = (row['ema8'] > row['ema21'])*(row['rsi'] < params['rsi_buy'])*(row['macd'] > row['macd_signal'])
    sell = (row['ema8'] < row['ema21'])*(row['rsi'] > params['rsi_sell'])*(row['macd'] < row['macd_signal'])
    if buy:
        return 1
    elif sell:
        return -1
    else:
        return 0

# ======================
# Funções de backtest e paper
# ======================
def simular_backtest(df, params):
    df = calcular_indicadores(df)
    capital = CAPITAL_INICIAL
    equity = [capital]
    trades = []
    for idx,row in df.iterrows():
        sinal = sinal_confidence_row(row, params)
        if sinal != 0:
            trade = {
                'time': str(row['datetime']),
                'sinal': sinal,
                'preco': row['close'],
                'capital_antes': capital
            }
            pnl = (row['close'] - trades[-1]['preco']) * sinal if trades else 0
            capital += pnl
            trade['capital_depois'] = capital
            trades.append(trade)
        equity.append(capital)
    lucro = capital - CAPITAL_INICIAL
    drawdown = max(0, max(equity) - min(equity))
    winrate = np.mean([1 if t['capital_depois']>t['capital_antes'] else 0 for t in trades]) if trades else 0
    profit_factor = (sum([t['capital_depois']-t['capital_antes'] for t in trades if t['capital_depois']>t['capital_antes']])+1e-9)/ (abs(sum([t['capital_depois']-t['capital_antes'] for t in trades if t['capital_depois']<t['capital_antes']]))+1e-9)
    rr = lucro / (drawdown+1e-9)
    return {'lucro': lucro, 'dd': drawdown, 'winrate': winrate, 'profit_factor': profit_factor, 'rr': rr, 'equity': equity, 'trades_details': trades, 'params': params}

def simular_paper(df, best_params):
    df = calcular_indicadores(df)
    capital = CAPITAL_INICIAL
    trades = []
    for idx,row in df.iterrows():
        sinal = sinal_confidence_row(row, best_params)
        if sinal != 0:
            trade = {'time': str(row['datetime']), 'sinal': sinal, 'preco': row['close'], 'capital_antes': capital}
            pnl = (row['close'] - trades[-1]['preco']) * sinal if trades else 0
            capital += pnl
            trade['capital_depois'] = capital
            trades.append(trade)
    save_paper_state('PAPER', capital, trades[-1] if trades else None)
    return trades

# ======================
# Funções auxiliares
# ======================
def salvar_grafico(equity, ativo):
    plt.figure(figsize=(10,4))
    plt.plot(equity)
    plt.title(f"Equity {ativo}")
    plt.savefig(os.path.join(PLOT_PATH, f"equity_{ativo}.png"))
    plt.close()

def gerar_params_aleatorios():
    return {
        'rsi_buy': random.randint(25,45),
        'rsi_sell': random.randint(55,75)
    }

# ======================
# Loop principal
# ======================
if __name__=="__main__":
    criar_tabela_db()
    while True:
        for ativo in ATIVOS:
            log(f"Processando {ativo}")
            df = carregar_candles(ativo)
            if df is None or df.empty:
                log(f"{ativo}: dados insuficientes")
                continue

            combinacoes = COMBINACOES_POR_ATIVO + (COMBINACOES_EXTRA_WINZ if ativo=="WINZ25" else 0)
            resultados = []
            for _ in range(combinacoes):
                params = gerar_params_aleatorios()
                res = simular_backtest(df, params)
                resultados.append(res)
            top1 = max(resultados, key=lambda x: x['lucro'])
            salvar_grafico(top1['equity'], ativo)
            save_or_replace_best(ativo, "M5", top1, top1['params'], top1['equity'], top1['trades_details'])
        log(f"Pausa {PAUSA_SEGUNDOS}s antes da próxima rodada")
        time.sleep(PAUSA_SEGUNDOS)

"""
pipeline_search_ensemble_ml_fixed.py

Versão corrigida e robusta do pipeline:
- corrige erro 'ema_8' garantindo cálculo de todas EMAs usadas
- leitura robusta de CSVs MT5 (DATE + TIME)
- wildcard para arquivos de contrato (WIN$, WINZ*, etc)
- otimização genética por ativo e ensemble
- salva resultados e curvas
"""

import os
import glob
import json
import random
import math
from itertools import product
from datetime import datetime
from datetime import timedelta
from collections import defaultdict
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import joblib

# ML
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# ---------------------------
# CONFIG
# ---------------------------
BASE_PATH = r"C:\Users\Marco\Documents\dados"
OUT_PATH = r"C:\Users\Marco\ia_finance"
MODELS_PATH = os.path.join(OUT_PATH, "modelos")
os.makedirs(OUT_PATH, exist_ok=True)

# If you want to specify explicit filenames, put them in the lists. (FIXED: use m5_all.py output)
# You may also supply globs like "WIN*.csv" or "WIN$*.csv" and the script will expand them.
SYMBOL_FILES = {
    "PETR4": ["PETR4.csv"],
    "WIN":   ["WIN$.csv"],
    "WDO":   ["WDO$.csv"]
}

ASSET_CONFIG = {
    # Values are examples. Adjust for your broker/market conditions.
    # commission: cost for a round-trip trade (buy + sell).
    # slippage_points: how many points you lose on average for a round-trip trade.
    "PETR4": {"point_value": 1,    "commission": 0.02, "slippage_points": 0.02}, # R$0.01 slippage, R$0.01 commission
    "WIN":   {"point_value": 0.2,  "commission": 1.00, "slippage_points": 10},   # 5 pts slippage, R$1.00 commission
    "WDO":   {"point_value": 10,   "commission": 3.00, "slippage_points": 1},    # 0.5 pts slippage, R$3.00 commission
}

ASSET_STRATEGIES = {
    "PETR4": ["pullback", "macross", "rsi_reversal", "bb_reversion"],
    "WIN":   ["macross", "pullback", "rsi_reversal"],
    "WDO": [
        "vwap_std_reversion",
        "vwap_rejection",
        "atr_breakout",
        "candle_reversal",
        "rsi_reversal",
        "bb_reversion"
    ]
}



CAPITAL = 100_000 # Capital inicial para cálculo de PnL
WF_TRAIN_MONTHS = 6 # Meses para treino no walk-forward
WF_TEST_MONTHS = 2 # Meses para teste no walk-forward
LOOKAHEAD_BARS = 20 # Barras para verificar TP/SL intrabar
MIN_TRADES = 30 # Regras com menos trades que isso são desconsideradas
N_GEN = 30 # Gerações do otimizador genético
POP_SIZE = 30 # Tamanho da população do otimizador genético
ELITE = 6 # Número de indivíduos de elite para a próxima geração
TOP_K_ENSEMBLE = 6
RANDOM_INJECT = 4 # Número de indivíduos aleatórios para injetar em cada nova geração para diversidade
EARLY_STOP_GENS = 5 # Parar se a melhor pontuação não melhorar por N gerações
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# parallel jobs
CPU_COUNT = max(1, multiprocessing.cpu_count() - 1) # Usar todos os cores menos um

# ---------------------------
# UTIL: carregar CSV MT5 e criar datetime (robusto)
# ---------------------------
def load_mt5_csv(path):
    # detect separator: try tab first, then comma
    try:
        df = pd.read_csv(path, sep='\t', engine='python')
        if df.shape[1] == 1:
            # maybe it was comma separated
            df = pd.read_csv(path, sep=',', engine='python')
    except Exception:
        df = pd.read_csv(path, sep=',', engine='python')

    # normalize column names (remove < > and lower)
    cols = [c.strip().lower().replace("<", "").replace(">", "") for c in df.columns]
    df.columns = cols

    # build datetime column
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors='coerce')
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
    else:
        raise ValueError(f"{path} doesn't have date/time columns")

    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep='first')]

    # normalize names for OHLC and others
    mapping = {}
    for col in ['open','high','low','close','tickvol','vol','spread']:
        if col in df.columns:
            mapping[col] = col
        else:
            for c in df.columns:
                if c.startswith(col[:3]):
                    mapping[c] = col
                    break
    df = df.rename(columns=mapping)

    # keep only needed columns (prefer open high low close; keep vol/tickvol/spread if present)
    keep = [c for c in ['open','high','low','close','tickvol','vol','spread'] if c in df.columns]
    if not {'open','high','low','close'}.issubset(set(keep)):
        raise ValueError(f"{path} missing required OHLC columns after normalization: {keep}")

    df = df[keep]
    return df

def expand_file_list(list_patterns):
    """
    From patterns list (globs or explicit filenames), return real files existing in BASE_PATH.
    """
    files = []
    for pat in list_patterns:
        if any(ch in pat for ch in ['*','?','$']):  # treat as glob
            glob_path = os.path.join(BASE_PATH, pat) # FIXED: use BASE_PATH for glob
            found = glob.glob(glob_path)
            found_sorted = sorted(found)
            files.extend(found_sorted)
        else:
            p = os.path.join(BASE_PATH, pat)
            if os.path.exists(p):
                files.append(p)
            else:
                # try glob with wildcard around
                alt = glob.glob(os.path.join(BASE_PATH, f"*{pat}*"))
                if alt:
                    files.extend(sorted(alt))
    # unique & sorted
    files = sorted(list(dict.fromkeys(files)))
    return files

def load_symbol_continuous(patterns_or_files, symbol_name): # FIXED: pass symbol_name
    files = expand_file_list(patterns_or_files)
    if not files:
        print(f"[WARN] nenhum arquivo encontrado para padrões: {patterns_or_files}")
        return None
    dfs = []
    for f in files:
        try:
            df = load_mt5_csv(f)
            # FIXED: filter by symbol_name if it's in the filename
            if symbol_name.lower() in os.path.basename(f).lower():
                dfs.append(df)
            else:
                # FIXED: if not matching, try to load the specific file from m5_all.py output
                specific_file = os.path.join(BASE_PATH, f"{symbol_name}_M5_REAL_2018-2024.csv")
                if os.path.exists(specific_file):
                    dfs.append(load_mt5_csv(specific_file))
            print(f"  → carregado {os.path.basename(f)}: {len(df):,} candles (M1?)")
        except Exception as e:
            print(f"  [WARN] falha ao ler {f}: {e}")
    if not dfs:
        return None
    df_all = pd.concat(dfs).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep='first')]
    print(f"  → total concatenado: {len(df_all):,} candles")
    return df_all

# ---------------------------
# Agregação M1 -> M5
# ---------------------------
def maybe_to_m5(df, symbol_name): # FIXED: pass symbol_name
    dif = df.index.to_series().diff().dropna()
    if len(dif) == 0:
        return df
    med_minutes = int(pd.Timedelta(dif.median()).seconds // 60)
    if med_minutes == 1: # if median diff is 1 minute, convert to 5min
        df5 = df.resample("5T").agg({
            'open':'first','high':'max','low':'min','close':'last',
            'tickvol': 'sum' if 'tickvol' in df.columns else 'last',
            'vol': 'sum' if 'vol' in df.columns else 'last',
        }).dropna()
        print(f"  → convertido para M5: {len(df5):,} candles")
        return df5
    else:
        print(f"  → assumindo timeframe atual (med {med_minutes} min)")
        return df

# ---------------------------
# Features (garante todas EMAs necessárias)
# ---------------------------
def add_features(df):
    df = df.copy()
    c = df['close']

    # spans usados pelo GA (inclui 5,8,9,13,21,34,50,89,100,200) (FIXED: ensure all EMAs are calculated)
    spans = [5,8,9,13,21,34,50,89,100,200]
    for span in spans:
        df[f'ema_{span}'] = c.ewm(span=span, adjust=False).mean()
        df[f'sma_{span}'] = c.rolling(span).mean()

    # RSI 14
    delta = c.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    rsi = 100 - 100/(1 + (up.rolling(14).mean() / (down.rolling(14).mean().replace(0, np.nan))))
    df['rsi_14'] = rsi

    # ATR 14
    tr = pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - c.shift(1)).abs(),
        (df['low'] - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()

    # vol
    df['vol'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['vol_med'] = df['vol'].rolling(50).mean()

    # --- IMPROVEMENT: Add daily VWAP ---
    # --- FIX: Make VWAP calculation robust by using 'vol' or 'tickvol' ---
    volume_col = 'vol' if 'vol' in df.columns else 'tickvol'
    if volume_col in df.columns:
        df['price_vol'] = (df['close'] * df[volume_col])
        df['cum_vol'] = df.groupby(df.index.date)[volume_col].cumsum()
        df['cum_price_vol'] = df.groupby(df.index.date)['price_vol'].cumsum()
        df['vwap'] = df['cum_price_vol'] / df['cum_vol']
        df.drop(columns=['price_vol', 'cum_vol', 'cum_price_vol'], inplace=True)
    else:
        print("  [WARN] VWAP not calculated: 'vol' or 'tickvol' column not found.")
    # --- IMPROVEMENT: Add features for new WDO strategies ---
    df['rolling_std_20'] = c.rolling(20).std()
    df['rolling_high_20'] = df['high'].rolling(20).max()
    df['rolling_low_20'] = df['low'].rolling(20).min()
    df['candle_range'] = df['high'] - df['low']

    # --- STRATEGY EXPANSION: Add Bollinger Bands ---
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    df['bb_mid'] = sma_20 # Can be used for exits or trend filter


    df = df.dropna()
    return df

# ---------------------------
# Simulação intrabar (com tratamento de TP/SL e ordem de ocorrência)
# ---------------------------
def is_petr4_market_hours(index):
    """Applies a more restrictive time filter for PETR4 to avoid opening/closing noise."""
    return (index.hour >= 10) & (index.minute >= 30) & (index.hour < 17)


def simulate_trades(df, signals, sl_atr_mult, tp_atr_mult, point_value, commission, slippage_points, size=1, lookahead=LOOKAHEAD_BARS, asset_name=""):
    # --- OPTIMIZATION: Extract numpy arrays outside the loop to avoid pandas overhead ---
    atr_values = df['atr_14'].to_numpy()
    open_prices = df['open'].to_numpy()
    high_prices = df['high'].to_numpy()
    low_prices = df['low'].to_numpy()
    close_prices = df['close'].to_numpy()
    
    # Align signals and convert to numpy array for faster access
    signals_aligned = signals.reindex(df.index, fill_value=0).to_numpy()

    pnl = []
    false_quick_stops = 0
    
    # The loop is now over numpy arrays, which is much faster than pandas .iloc
    for i in range(len(df) - lookahead - 2):
        direction = signals_aligned[i]
        if direction == 0:
            continue
        
        atr_val = atr_values[i]
        entrada = open_prices[i+1]
        
        # Slicing numpy arrays is faster than pandas
        highs = high_prices[i+1:i+1+lookahead]
        lows = low_prices[i+1:i+1+lookahead]

        # --- IMPROVEMENT: Handle both LONG and SHORT trades ---
        # --- IMPROVEMENT: ATR-based SL/TP ---
        sl_pts_asset = sl_atr_mult * atr_val
        tp_pts_asset = tp_atr_mult * atr_val

        if direction == 1: # Long trade
            hit_tp = np.where(highs >= entrada + tp_pts_asset)[0]
            hit_sl = np.where(lows <= entrada - sl_pts_asset)[0]
        else: # Short trade
            hit_tp = np.where(lows <= entrada - tp_pts_asset)[0]
            hit_sl = np.where(highs >= entrada + sl_pts_asset)[0]

        pnl_pts = None
        if hit_tp.size > 0 and hit_sl.size > 0:
            tp_idx = int(hit_tp[0]); sl_idx = int(hit_sl[0])
            if tp_idx < sl_idx:
                pnl_pts = tp_pts_asset
            else:
                pnl_pts = -sl_pts_asset
            if (pnl_pts < 0) and sl_idx <= 3:
                false_quick_stops += 1
        elif hit_tp.size > 0:
            pnl_pts = tp_pts_asset
        elif hit_sl.size > 0:
            pnl_pts = -sl_pts_asset
            if 0 <= int(hit_sl[0]) <= 3:
                false_quick_stops += 1
        else:
            fechamento = close_prices[i+1+lookahead-1]
            pnl_pts = (fechamento - entrada) * direction # PnL in points

        # --- REALISM IMPROVEMENT: Subtract Costs ---
        # --- FIX: Correctly apply costs to all outcomes ---
        pnl_gross_money = pnl_pts * point_value * size
        slippage_cost = slippage_points * point_value * size
        total_cost = slippage_cost + commission
        pnl_net_money = pnl_gross_money - total_cost
        pnl.append(pnl_net_money)
    return pnl, false_quick_stops

# ---------------------------
# Métricas e score
# ---------------------------
def metrics_from_pnl(pnl):
    if not pnl:
        return None
    arr = np.array(pnl)
    profit = float(arr.sum())
    wins = float(arr[arr>0].sum()) if arr[arr>0].size>0 else 0.0
    losses = float(-arr[arr<0].sum()) if arr[arr<0].size>0 else 0.0
    pf = float(wins / (losses or 1.0))
    winrate = float((arr>0).sum()/len(arr) * 100)
    eq = np.cumsum(arr) + CAPITAL
    peak = np.maximum.accumulate(eq)
    maxdd = float(np.max(peak - eq))

    # --- FIX: Calculate Sharpe Ratio ---
    returns_std = arr.std() if len(arr) > 1 else 1.0
    sharpe = (arr.mean() / (returns_std or 1.0)) * math.sqrt(252)

    # --- IMPROVEMENT: Calculate Sortino Ratio for better risk-adjusted return metric ---
    downside_returns = arr[arr < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 1 else 1.0
    sortino = (arr.mean() / (downside_std or 1.0)) * math.sqrt(252)

    return {
        "profit": profit,
        "sharpe": round(sharpe, 3),
        "trades": int(len(arr)),
        "winrate": winrate,
        "pf": round(pf,3),
        "maxdd": round(maxdd,2),
        "sortino": round(sortino,3)
    }

def penalized_score(metrics, false_quick_stops, months_no_trade):
    # --- IMPROVEMENT: More robust score with penalties for inactivity and instability ---
    if not metrics or metrics['trades'] < MIN_TRADES:
        return -1e9 # Severe penalty for not enough trades

    # The primary driver is the risk-adjusted return (Sortino)
    score = metrics.get('sortino', -1e9) * 1000
    if np.isnan(score):
        return -1e9

    # 1. Penalty for instability (quick stops)
    quick_stop_ratio = false_quick_stops / metrics['trades'] if metrics['trades'] > 0 else 0
    # Penalize more heavily as the ratio of bad entries increases
    score *= (1 - (quick_stop_ratio * 2))

    # 2. Penalty for inactivity
    score -= months_no_trade * 50 # Apply a fixed penalty for each month without trades

    return score

# ---------------------------
# Strategy families generation (rule-based)
# ---------------------------
def generate_rule_population(asset_name, seed_pop=None, count=POP_SIZE):
    pop = []
    families = ASSET_STRATEGIES.get(asset_name, ['macross', 'volbreak']) # Default if asset not in config
    if seed_pop:
        pop.extend(seed_pop) # FIXED: extend with seed_pop
    while len(pop) < count:
        f = random.choice(families)
        if f == 'pullback':
            p = {'vol_mult': random.choice([1.1, 1.5, 2.0])}
            sl_atr, tp_atr = random.choice([(1.5, 2.5), (2.0, 4.0)])
        elif f == 'macross':
            fast = random.choice([5,8,9,13,21])
            slow = random.choice([21,34,50,89])
            if fast >= slow: fast = max(5, slow-5)
            p = {'fast':fast,'slow':slow}
            sl_atr, tp_atr = random.choice([(2.0, 3.0), (2.5, 5.0)])
        elif f == 'volbreak':
            lookback = random.choice([10,15,20,30])
            mult = random.choice([1.1,1.3,1.5,1.8])
            p = {'lookback':lookback,'mult':mult}
            sl_atr, tp_atr = random.choice([(2.0, 3.0), (3.0, 5.0)])
        elif f == 'reversal':
            p = {'rsi_level': random.choice([20, 25, 30, 35])}
            sl_atr, tp_atr = random.choice([(2.0, 4.0), (3.0, 6.0)])
        elif f == 'vwap_bounce':
            p = {} # No parameters needed for basic version
            sl_atr, tp_atr = random.choice([(1.5, 3.0), (2.0, 4.0)])

        # --- WDO-Specific Strategies ---
        elif f == 'vwap_std_reversion':
            p = {'std_mult': random.choice([1.0, 1.5, 2.0])}
            sl_atr, tp_atr = random.choice([(1.0, 1.5), (1.5, 2.5)])
        elif f == 'vwap_rejection':
            p = {}
            sl_atr, tp_atr = random.choice([(1.2, 1.8), (1.5, 2.5)])
        elif f == 'atr_breakout':
            p = {'atr_mult': random.choice([0.5, 0.8, 1.0])}
            sl_atr, tp_atr = random.choice([(2.0, 3.0), (2.5, 4.0)])
        elif f == 'candle_reversal':
            p = {'body_pct': random.choice([0.2, 0.3])} # Body must be in top/bottom 20-30% of range
            sl_atr, tp_atr = random.choice([(1.5, 2.0), (2.0, 3.0)])

        # --- STRATEGY EXPANSION: New Strategies ---
        elif f == 'bb_reversion':
            p = {'std_mult': random.choice([1.8, 2.0, 2.2])} # Parameters for BB
            sl_atr, tp_atr = random.choice([(1.5, 2.0), (2.0, 3.5)])
        elif f == 'rsi_reversal':
            p = {'buy_level': random.choice([20,25,30]), 'sell_level': random.choice([70,75,80])}
            sl_atr, tp_atr = random.choice([(2.0, 4.0), (2.5, 5.0)])
        
        # --- IMPROVEMENT: Allow GA to explore long, short, or bi-directional for most strategies ---
        bi_directional_strategies = ['vwap_std_reversion', 'vwap_rejection', 'candle_reversal', 'bb_reversion', 'rsi_reversal', 'atr_breakout']
        
        # For all strategies, let the GA decide the direction (1=long, -1=short, 0=bi-directional)
        # The signal_from_strategy function will handle the actual signal generation based on this.
        direction = random.choice([1, -1, 0])

        pop.append({'family':f, 'params':p, 'sl_atr':sl_atr, 'tp_atr':tp_atr, 'direction': direction})
    return pop

# ---------------------------
# Strategy signals (com defensivas)
# ---------------------------
def ensure_ema_exists(df, span):
    col = f'ema_{span}'
    if col not in df.columns:
        df[col] = df['close'].ewm(span=span, adjust=False).mean()

def is_wdo_market_hours(index):
    """Applies the time filter for WDO strategies."""
    return ((index.hour >= 9) & (index.minute >= 5) & (index.hour < 12)) | \
           ((index.hour >= 14) & (index.hour < 17))

def signal_from_strategy(df, ind):
    # --- MAJOR REFACTOR: Generate long/short signals based on direction ---
    fam = ind['family']; p = ind['params']
    # defensive: ensure EMAs needed exist
    if fam == 'macross':
        for s in [p.get('fast'), p.get('slow')]:
            if s is not None:
                ensure_ema_exists(df, s)

    # compute signals
    # --- IMPROVEMENT: Add specific time filter for PETR4 strategies ---
    is_petr4 = 'PETR4' in df.attrs.get('asset_name', '')
    time_filter = is_petr4_market_hours(df.index) if is_petr4 else True

    direction = ind.get('direction', 1)
    buy_signal = pd.Series(False, index=df.index)
    sell_signal = pd.Series(False, index=df.index)

    if fam == 'pullback':
        # defensive ensure
        for s in [21,50,200]: # FIXED: ensure all EMAs are calculated
            ensure_ema_exists(df, s)
        
        if direction in [1, 0]: # Long signal
            buy_signal = (df['close'] < df['ema_21']) & (df['close'].shift(1) >= df['ema_21'].shift(1))
            buy_signal &= (df['vol'] > df['vol_med'] * p['vol_mult'])
            buy_signal &= (df['close'] > df['ema_50']) & (df['ema_50'] > df['ema_200'])
        if direction in [-1, 0]: # Short signal
            sell_signal = (df['close'] > df['ema_21']) & (df['close'].shift(1) <= df['ema_21'].shift(1))
            sell_signal &= (df['vol'] > df['vol_med'] * p['vol_mult'])
            sell_signal &= (df['close'] < df['ema_50']) & (df['ema_50'] < df['ema_200'])
        
        signal = pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)
        signal &= time_filter
        return signal

    elif fam == 'macross':
        fast = p['fast']; slow = p['slow']
        if direction in [1, 0]: # Golden Cross
            buy_signal = (df[f'ema_{fast}'] > df[f'ema_{slow}']) & (df[f'ema_{fast}'].shift(1) <= df[f'ema_{slow}'].shift(1))
        if direction in [-1, 0]: # Death Cross
            sell_signal = (df[f'ema_{fast}'] < df[f'ema_{slow}']) & (df[f'ema_{fast}'].shift(1) >= df[f'ema_{slow}'].shift(1))
        
        signal = pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)
        signal &= time_filter
        return signal

    elif fam == 'volbreak':
        lookback = p['lookback']; mult = p['mult']
        rolling_std = df['close'].pct_change().rolling(lookback).std()
        threshold = rolling_std * mult
        candle_move = (df['close'] - df['open']).abs() / df['open'].shift(1)
        is_break = (candle_move > threshold) & (df['vol'] > df['vol_med']*0.8)
        
        if direction in [1, 0]: # Long on positive candle break
            buy_signal = is_break & (df['close'] > df['open'])
        if direction in [-1, 0]: # Short on negative candle break
            sell_signal = is_break & (df['close'] < df['open'])

        signal = pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)
        signal &= time_filter
        return signal

    elif fam == 'reversal':
        rsi_level = p['rsi_level']
        if direction in [1, 0]: # Buy on RSI oversold exit
            buy_signal = (df['rsi_14'] < rsi_level) & (df['rsi_14'].shift(1) >= rsi_level)
        if direction in [-1, 0]: # Sell on RSI overbought exit
            sell_signal = (df['rsi_14'] > (100 - rsi_level)) & (df['rsi_14'].shift(1) <= (100 - rsi_level))
        
        signal = pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)
        signal &= time_filter
        return signal

    elif fam == 'vwap_bounce':
        # --- FIX: Respect direction parameter ---
        if direction in [1, 0]:
            buy_signal = (df['close'] > df['vwap']) & (df['close'].shift(1) <= df['vwap'].shift(1))
        if direction in [-1, 0]:
            sell_signal = (df['close'] < df['vwap']) & (df['close'].shift(1) >= df['vwap'].shift(1))
        
        signal = pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)
        signal &= time_filter
        return signal




    # --- WDO-Specific Strategy Logic ---
    elif fam == 'vwap_std_reversion':
        if 'vwap' not in df.columns: return pd.Series(0, index=df.index) # Defensive check
        std_mult = p['std_mult']
        
        if direction in [1, 0]:
            buy_signal = df['close'] < (df['vwap'] - std_mult * df['rolling_std_20'])
        if direction in [-1, 0]:
            sell_signal = df['close'] > (df['vwap'] + std_mult * df['rolling_std_20'])
        
        signal = pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)
        signal &= is_wdo_market_hours(df.index)
        return signal

    elif fam == 'vwap_rejection':
        if 'vwap' not in df.columns: return pd.Series(0, index=df.index) # Defensive check
        is_neg_candle = df['close'] < df['open']
        is_pos_candle = df['close'] > df['open']
        
        if direction in [1, 0]:
            buy_signal = (df['close'] < df['vwap']) & is_pos_candle
        if direction in [-1, 0]:
            sell_signal = (df['close'] > df['vwap']) & is_neg_candle
        
        signal = pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)
        signal &= is_wdo_market_hours(df.index)
        return signal

    elif fam == 'atr_breakout':
        atr_mult = p['atr_mult']
        # --- FIX: Make ATR Breakout bi-directional ---
        if direction in [1, 0]:
            buy_signal = df['close'] > (df['rolling_high_20'] + atr_mult * df['atr_14'])
        if direction in [-1, 0]:
            sell_signal = df['close'] < (df['rolling_low_20'] - atr_mult * df['atr_14'])
        
        signal = pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)
        signal &= is_wdo_market_hours(df.index)
        return signal

    elif fam == 'candle_reversal':
        body_pct = p['body_pct']
        
        if direction in [1, 0]:
            # Bullish reversal: new low, but closes in top % of its range (pinbar/hammer)
            buy_signal = (df['low'] < df['low'].shift(1)) & (df['close'] > (df['high'] - df['candle_range'] * body_pct))
        if direction in [-1, 0]:
            # Bearish reversal: new high, but closes in bottom % of its range (shooting star)
            sell_signal = (df['high'] > df['high'].shift(1)) & (df['close'] < (df['low'] + df['candle_range'] * body_pct))
        
        signal = pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)
        signal &= is_wdo_market_hours(df.index)
        return signal

    # --- STRATEGY EXPANSION: New Strategy Logic ---
    elif fam == 'bb_reversion':
        std_mult = p.get('std_mult', 2.0) # Use 2.0 as default if not specified
        upper_band = df['bb_mid'] + (df['rolling_std_20'] * std_mult)
        lower_band = df['bb_mid'] - (df['rolling_std_20'] * std_mult)
        
        if direction in [1, 0]:
            buy_signal = (df['close'] < lower_band) & (df['close'].shift(1) >= lower_band.shift(1))
        if direction in [-1, 0]:
            sell_signal = (df['close'] > upper_band) & (df['close'].shift(1) <= upper_band.shift(1))
        
        signal = pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)
        signal &= time_filter
        return signal

    elif fam == 'rsi_reversal':
        buy_level = p.get('buy_level', 30)
        sell_level = p.get('sell_level', 70)
        
        if direction in [1, 0]:
            buy_signal = (df['rsi_14'] < buy_level) & (df['rsi_14'].shift(1) >= buy_level)
        if direction in [-1, 0]:
            sell_signal = (df['rsi_14'] > sell_level) & (df['rsi_14'].shift(1) <= sell_level)
        
        signal = pd.Series(np.where(buy_signal, 1, np.where(sell_signal, -1, 0)), index=df.index)
        signal &= time_filter
        return signal
    else:
        return pd.Series(0, index=df.index)

# ---------------------------
# Walk-forward evaluation
# ---------------------------
def months_between(a,b):
    return (b.year - a.year)*12 + (b.month - a.month)

def walkforward_evaluate(df_full, indiv, asset_name):
    start = df_full.index.min()
    end = df_full.index.max()
    cur = start
    all_pnl = [] # --- IMPROVEMENT: Concatenate PnL from all folds for a single robust evaluation ---
    months_no_trade = 0
    total_false_quick_stops = 0 # --- IMPROVEMENT: Track quick stops across folds ---

    while True:
        train_end = (cur + pd.DateOffset(months=WF_TRAIN_MONTHS)) - pd.Timedelta(seconds=1)
        test_end = train_end + pd.DateOffset(months=WF_TEST_MONTHS)
        if test_end > end:
            break
        train = df_full[cur:train_end].copy()
        test = df_full[train_end + pd.Timedelta(seconds=1):test_end].copy()
        cur = cur + pd.DateOffset(months=WF_TEST_MONTHS)
        if len(train) < 50 or len(test) < 10:
            continue
        sig = signal_from_strategy(test, indiv)
        
        # Use new ASSET_CONFIG for costs
        cfg = ASSET_CONFIG.get(asset_name, ASSET_CONFIG['PETR4']) # Default to PETR4 if not found
        
        # --- IMPROVEMENT: Use longer lookahead for PETR4 trend strategies ---
        lookahead_bars = LOOKAHEAD_BARS
        is_petr4_trend = asset_name == 'PETR4' and indiv['family'] in ['pullback', 'macross']
        if is_petr4_trend:
            lookahead_bars = 40 # 200 minutes, allows more time for trend to develop

        pnl, false_quick = simulate_trades(test, sig, indiv['sl_atr'], indiv['tp_atr'], cfg['point_value'], cfg['commission'], cfg['slippage_points'], asset_name=asset_name, lookahead=lookahead_bars)

        if not pnl:
            months_no_trade += months_between(test.index.min(), test.index.max())
        
        all_pnl.extend(pnl)
        total_false_quick_stops += false_quick

    # --- IMPROVEMENT: Calculate metrics on the combined out-of-sample PnL ---
    final_metrics = metrics_from_pnl(all_pnl)
    score = penalized_score(final_metrics, total_false_quick_stops, months_no_trade)
    return final_metrics, score

# ---------------------------
# Genetic-like optimization
# ---------------------------
def evolve_population(df, asset_name, seed_pop=None): # Pass asset_name
    pop = generate_rule_population(asset_name, seed_pop=seed_pop) # Pass asset_name
    evaluated = []
    best_overall_score = -1e10
    gens_no_improve = 0

    for gen in range(N_GEN):
        results = Parallel(n_jobs=CPU_COUNT)(delayed(walkforward_evaluate)(df, ind, asset_name) for ind in pop)
        
        # --- FIX: Process results robustly, handling None from evaluation ---
        scored = []
        for ind, (metrics, score) in zip(pop, results):
            scored.append((score, ind, metrics))

        # Sort by the new robust score
        scored.sort(key=lambda x: x[0], reverse=True)
        elite = [x[1] for x in scored[:ELITE]]
        evaluated.extend(scored[:ELITE])

        best = scored[0] # Best is now the one with the highest Sortino-based score
        best_score = best[0]
        best_ind = best[1]
        best_res = best[2]
        print(f"[EVO {asset_name}] GEN {gen+1}/{N_GEN} best score {best_score:.1f} family {best_ind['family']} params {best_ind['params']} trades {best_res['trades'] if best_res else 0}")

        # --- IMPROVEMENT: Early Stopping ---
        if best_score > best_overall_score:
            best_overall_score = best_score
            gens_no_improve = 0
        else:
            gens_no_improve += 1
        if gens_no_improve >= EARLY_STOP_GENS:
            print(f"  → Early stopping: best score hasn't improved in {EARLY_STOP_GENS} generations.")
            break

        newpop = elite.copy()
        # if elite smaller than needed, refill with random
        # --- IMPROVEMENT: Inject random individuals to maintain diversity ---
        random_inds_to_add = min(RANDOM_INJECT, POP_SIZE - len(newpop))
        if random_inds_to_add > 0:
            newpop.extend(generate_rule_population(asset_name, count=random_inds_to_add))

        # --- IMPROVEMENT: Use Tournament Selection for Crossover to increase diversity ---
        def tournament_selection(population, k=3):
            # Select k individuals from the population at random
            tournament_contenders = random.sample(population, k)
            # The winner is the one with the best (highest) score
            winner = max(tournament_contenders, key=lambda x: x[0])
            return winner[1] # Return the individual dict

        while len(newpop) < POP_SIZE:
            # Select parents from the entire scored population, not just the elite
            if len(scored) >= 3:
                p1 = tournament_selection(scored, k=3)
                p2 = tournament_selection(scored, k=3)
                child = crossover(p1, p2)
            else:
                child = generate_rule_population(asset_name, count=1)[0]
            child = mutate(child, asset_name)
            newpop.append(child)
        pop = newpop
    evaluated_sorted = sorted(evaluated, key=lambda x: x[0], reverse=True)
    top = []
    for score, ind, res in evaluated_sorted:
        if len(top) >= TOP_K_ENSEMBLE:
            break
        if ind is None: 
            continue
        key = (ind['family'], json.dumps(ind['params'], sort_keys=True), ind['sl_atr'], ind['tp_atr'])
        if key in [ (t['family'], json.dumps(t['params'], sort_keys=True), t['sl_atr'], t['tp_atr']) for t in top ]:
            continue
        top.append(ind)
    return top

def crossover(a,b):
    # --- IMPROVEMENT: Make crossover more intelligent to avoid creating invalid individuals ---
    # The child inherits the family and base parameters from one parent (p1).
    p1, p2 = (a, b) if random.random() < 0.5 else (b, a)
    child = {
        'family': p1['family'],
        'params': p1['params'].copy(), # Start with a copy of the first parent's params
        'sl_atr': (p1['sl_atr'] + p2['sl_atr']) / 2 if random.random() < 0.7 else p1['sl_atr'],
        'tp_atr': (p1['tp_atr'] + p2['tp_atr']) / 2 if random.random() < 0.7 else p1['tp_atr'],
        'direction': p1['direction'] # Default to p1's direction
    }

    # Crossover direction with a small probability
    if random.random() < 0.2: # 20% chance to crossover direction
        child['direction'] = p2['direction']

    # Crossover parameters that are common to both parents
    for k in child['params']:
        if k in p2['params'] and random.random() < 0.5:
            if isinstance(child['params'][k], int):
                child['params'][k] = int(round((p1['params'][k] + p2['params'][k]) / 2))
            else: # float
                child['params'][k] = (p1['params'][k] + p2['params'][k]) / 2

    # Specific logic for macross to ensure fast < slow
    if child['family'] == 'macross':
        if child['params']['fast'] >= child['params']['slow']:
            # If crossover made them invalid, pick from one parent or re-generate
            if random.random() < 0.5:
                child['params']['fast'] = p1['params']['fast']
                child['params']['slow'] = p1['params']['slow']
            else:
                child['params']['fast'] = p2['params']['fast']
                child['params']['slow'] = p2['params']['slow']
            # Ensure fast < slow again if still an issue
            if child['params']['fast'] >= child['params']['slow']:
                child['params']['fast'] = max(5, child['params']['slow'] - random.randint(5, 15)) # Ensure a gap
                child['params']['slow'] = max(21, child['params']['fast'] + random.randint(5, 15))

    # Specific logic for rsi_reversal to ensure buy_level < sell_level
    if child['family'] == 'rsi_reversal':
        if child['params']['buy_level'] >= child['params']['sell_level']:
            # If crossover made them invalid, pick from one parent or re-generate
            if random.random() < 0.5:
                child['params']['buy_level'] = p1['params']['buy_level']
                child['params']['sell_level'] = p1['params']['sell_level']
            else:
                child['params']['buy_level'] = p2['params']['buy_level']
                child['params']['sell_level'] = p2['params']['sell_level']
            # Ensure buy_level < sell_level again
            if child['params']['buy_level'] >= child['params']['sell_level']:
                child['params']['buy_level'] = random.randint(20, 30)
                child['params']['sell_level'] = random.randint(70, 80)
    
    # Ensure SL/TP are positive and TP > SL
    child['sl_atr'] = round(max(0.5, child['sl_atr']), 2)
    child['tp_atr'] = round(max(child['sl_atr'] + 0.5, child['tp_atr']), 2) # TP must be greater than SL
    return child

def mutate(ind, asset_name):
    # --- IMPROVEMENT: Use higher mutation rate for PETR4 to increase exploration ---
    param_mutation_prob = 0.4 if asset_name == 'PETR4' else 0.2
    sltp_mutation_prob = 0.3 if asset_name == 'PETR4' else 0.15
    if random.random() < param_mutation_prob:
        if ind['family']=='pullback' and 'vol_mult' in ind['params']:
            ind['params']['vol_mult'] = max(1.05, ind['params']['vol_mult'] + random.choice([-0.2,-0.1,0.1,0.2]))
        elif ind['family']=='macross' and 'fast' in ind['params'] and 'slow' in ind['params']:
            # --- FIX: Ensure fast < slow after mutation ---
            new_fast = max(3, ind['params']['fast'] + random.choice([-2, -1, 1, 2]))
            new_slow = max(5, ind['params']['slow'] + random.choice([-3, -1, 1, 3]))
            if new_fast >= new_slow:
                new_slow = new_fast + random.randint(5, 15) # Ensure a valid gap if mutation fails
            ind['params']['fast'], ind['params']['slow'] = new_fast, new_slow
        elif ind['family']=='reversal' and 'rsi_level' in ind['params']:
            ind['params']['rsi_level'] = max(15, min(40, ind['params']['rsi_level'] + random.choice([-5, -2, 2, 5])))
        elif ind['family']=='volbreak' and 'lookback' in ind['params'] and 'mult' in ind['params']:
            ind['params']['lookback'] = max(5, ind['params']['lookback'] + random.choice([-5,-2,2,5]))
            ind['params']['mult'] = max(0.8, ind['params']['mult'] + random.choice([-0.2,-0.1,0.1,0.2]))
        # --- FIX: Add mutation logic for new WDO strategies ---
        elif ind['family']=='vwap_std_reversion' and 'std_mult' in ind['params']:
            ind['params']['std_mult'] = max(0.5, ind['params']['std_mult'] + random.choice([-0.5, -0.2, 0.2, 0.5]))
        elif ind['family']=='atr_breakout' and 'atr_mult' in ind['params']:
            ind['params']['atr_mult'] = max(0.2, ind['params']['atr_mult'] + random.choice([-0.3, -0.1, 0.1, 0.3]))
        elif ind['family']=='candle_reversal' and 'body_pct' in ind['params']:
            ind['params']['body_pct'] = max(0.1, min(0.5, ind['params']['body_pct'] + random.choice([-0.1, 0.1])))
        # --- STRATEGY EXPANSION: Mutation for new strategies ---
        elif ind['family']=='bb_reversion' and 'std_mult' in ind['params']:
            ind['params']['std_mult'] = max(1.5, ind['params']['std_mult'] + random.choice([-0.2, 0.2]))
        elif ind['family']=='rsi_reversal' and 'buy_level' in ind['params'] and 'sell_level' in ind['params']:
            ind['params']['buy_level'] = max(15, min(40, ind['params']['buy_level'] + random.choice([-5, 5])))
            ind['params']['sell_level'] = max(60, min(85, ind['params']['sell_level'] + random.choice([-5, 5])))

    if random.random() < sltp_mutation_prob:
        ind['sl_atr'] = max(0.5, ind['sl_atr'] * random.choice([0.9, 1.1]))
        ind['tp_atr'] = max(0.5, ind['tp_atr'] * random.choice([0.9, 1.1]))
        # Ensure TP > SL
        ind['tp_atr'] = max(ind['sl_atr'] + 0.5, ind['tp_atr'])

    # --- IMPROVEMENT: Allow direction mutation for most strategies ---
    if random.random() < 0.1: # 10% chance to change direction for any strategy
        ind['direction'] = random.choice([1, -1, 0]) # Can become uni-directional or stay bi-directional
    return ind

# ---------------------------
# Build ensemble from top strategies
# ---------------------------
def build_ensemble(df, top_inds, asset_name):
    if not top_inds:
        return None
    infos = []
    for ind in top_inds:
        sig = signal_from_strategy(df, ind)
        cfg = ASSET_CONFIG.get(asset_name, ASSET_CONFIG['PETR4'])
        pnl, false_quick = simulate_trades(df, sig, ind['sl_atr'], ind['tp_atr'], cfg['point_value'], cfg['commission'], cfg['slippage_points'], asset_name=asset_name)
        mets = metrics_from_pnl(pnl)
        if not mets: continue
        weight = max(0.0, mets['profit']) * max(0.1, mets['sharpe'])
        infos.append({'ind':ind,'pnl':pnl,'metrics':mets,'weight':weight,'signal':sig})
    if not infos:
        return None
    total_w = sum(i['weight'] for i in infos)
    for i in infos:
        i['norm_w'] = i['weight']/total_w if total_w>0 else 1.0/len(infos)
    sig_df = pd.DataFrame({f"s{i}":inf['signal'].astype(int) for i,inf in enumerate(infos)}, index=df.index)
    weights = np.array([inf['norm_w'] for inf in infos])
    weighted = sig_df.values.dot(weights)
    ensemble_signal = pd.Series(np.sign(weighted) * (np.abs(weighted) > 0.5), index=df.index, dtype=int)

    # --- IMPROVEMENT: Optimize SL/TP specifically for the ensemble signal ---
    cfg = ASSET_CONFIG.get(asset_name, ASSET_CONFIG['PETR4'])
    best_score = -1e10
    best_sl_atr, best_tp_atr = (2.0, 4.0) # Default values
    
    # Define a search space for SL/TP ATR multipliers
    sl_atr_options = [1.0, 1.5, 2.0, 2.5, 3.0]
    tp_atr_options = [1.5, 2.0, 3.0, 4.0, 5.0]

    for sl_mult, tp_mult in product(sl_atr_options, tp_atr_options):
        if tp_mult <= sl_mult: continue
        pnl, fq = simulate_trades(df, ensemble_signal, sl_mult, tp_mult, cfg['point_value'], cfg['commission'], cfg['slippage_points'], asset_name=asset_name)
        mets = metrics_from_pnl(pnl)
        score = penalized_score(mets, fq, 0)
        if score > best_score:
            best_score = score
            best_sl_atr, best_tp_atr = sl_mult, tp_mult

    pnl_ens, fq = simulate_trades(df, ensemble_signal, best_sl_atr, best_tp_atr, cfg['point_value'], cfg['commission'], cfg['slippage_points'], asset_name=asset_name)
    mets = metrics_from_pnl(pnl_ens)
    # --- FIX: Store the optimized SL/TP multipliers in the metrics dict ---
    if mets:
        mets['best_sl_atr'] = best_sl_atr
        mets['best_tp_atr'] = best_tp_atr
    return {'infos':infos, 'ensemble_signal':ensemble_signal, 'pnl':pnl_ens, 'metrics':mets}

# ---------------------------
# ML dataset & training
# ---------------------------
def prepare_ml_dataset(df, ensemble_signal, horizon=6):
    df2 = df.copy()
    feats = ['close','ema_9','ema_21','ema_50','rsi_14','atr_14','vol','vol_med']
    for f in feats:
        if f not in df2.columns:
            # compute fallback if possible
            if f.startswith('ema_'):
                span = int(f.split('_')[1])
                df2[f] = df2['close'].ewm(span=span, adjust=False).mean()
            else:
                df2[f] = df2['close']
    df2['ens_sig'] = ensemble_signal.astype(int)
    future_close = df2['close'].shift(-horizon)
    df2['label'] = (future_close > df2['close']).astype(int)
    df2 = df2.dropna()
    X = df2[feats + ['ens_sig']].copy()
    y = df2['label'].copy()
    return X, y, df2

def train_lgbm_walkforward(X, y, symbol):
    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    aucs = []
    all_preds = pd.Series(index=X.index, dtype=float)
    final_model = None
    features = list(X.columns)

    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        
        # IMPROVEMENT: Use a more powerful model like LightGBM
        model = lgb.LGBMClassifier(random_state=SEED, n_jobs=1)
        
        model.fit(Xtr, ytr)
        final_model = model # The last model in the loop is the one trained on the most recent data
        ypred = model.predict_proba(Xte)[:, 1] # Get probability of positive class
        
        # --- IMPROVEMENT: Store predictions for ML-filtered signal ---
        all_preds.iloc[test_idx] = ypred

        auc = roc_auc_score(yte, ypred) if len(np.unique(yte)) > 1 else 0.5
        models.append(model); aucs.append(float(auc))

    # --- SAVE THE FINAL MODEL AND FEATURES ---
    if final_model:
        os.makedirs(MODELS_PATH, exist_ok=True)
        model_filename = os.path.join(MODELS_PATH, f"{symbol}_lgbm_model.joblib")
        features_filename = os.path.join(MODELS_PATH, f"{symbol}_features.json")
        joblib.dump(final_model, model_filename)
        with open(features_filename, 'w') as f:
            json.dump({'features': features}, f, indent=2)
        print(f"  → Modelo final e features salvos para {symbol}.")

    return models, aucs, all_preds.dropna()

# ---------------------------
# Main pipeline per symbol
# ---------------------------
def run_for_symbol(symbol, patterns_or_files):
    # FIXED: Update SYMBOL_FILES to use the output of m5_all.py
    print(f"\n===== PROCESSANDO {symbol} =====")
    df_raw = load_symbol_continuous(patterns_or_files, symbol) # FIXED: pass symbol to load_symbol_continuous
    if df_raw is None or len(df_raw) < 200:
        print(f"[ERRO] dados insuficientes para {symbol} (len={0 if df_raw is None else len(df_raw)})")
        return None
    df = maybe_to_m5(df_raw, symbol) # FIXED: pass symbol to maybe_to_m5
    df = add_features(df)
    df.attrs['asset_name'] = symbol # Store asset name for later use

    # --- IMPROVEMENT: Load Hall of Fame to seed the population ---
    hall_of_fame_file = os.path.join(OUT_PATH, f"{symbol}_hall_of_fame.json")
    seed_pop = None
    if os.path.exists(hall_of_fame_file):
        try:
            with open(hall_of_fame_file, 'r') as f:
                loaded_pop = json.load(f)
            
            active_strategies = ASSET_STRATEGIES.get(symbol, [])
            
            # <<< CORREÇÃO AQUI: adicionar 'direction' se não existir >>>
            seed_pop = []
            for ind in loaded_pop:
                if ind.get('family') not in active_strategies:
                    continue
                # Garante que todo indivíduo carregado tenha 'direction'
                if 'direction' not in ind:
                    # Estratégias antigas eram só long → default = 1
                    ind['direction'] = 1
                # Garante que sl_atr e tp_atr existam
                ind.setdefault('sl_atr', 2.0)
                ind.setdefault('tp_atr', 4.0)
                seed_pop.append(ind)

            print(f"  → Semeando com {len(seed_pop)} indivíduos do 'hall of fame' (compatibilizados).")
        except Exception as e:
            print(f"  [WARN] Falha ao carregar hall of fame: {e}")
            seed_pop = None

    top_inds = evolve_population(df, symbol, seed_pop=seed_pop)
    ensemble = build_ensemble(df, top_inds, symbol)
    if top_inds:
        with open(hall_of_fame_file, 'w') as f:
            json.dump(top_inds, f, indent=2)

    ens_sig = ensemble['ensemble_signal'] if ensemble else pd.Series(False, index=df.index)
    X, y, df_ml = prepare_ml_dataset(df, ens_sig, horizon=6)
    if len(X) < 200:
        print("[WARN] Dados insuficientes para treinar ML. Usando apenas ensemble.")
        models, aucs, ml_preds = [], [], pd.Series()
        final_metrics = ensemble['metrics'] if ensemble else None
        final_pnl = ensemble['pnl'] if ensemble else []
    else:
        models, aucs, ml_preds = train_lgbm_walkforward(X, y, symbol)

        # --- IMPROVEMENT: Optimize ML confidence threshold ---
        best_ml_threshold = 0.5
        best_ml_filtered_score = -1e10
        
        cfg = ASSET_CONFIG.get(symbol, ASSET_CONFIG['PETR4'])
        sl_ens_atr, tp_ens_atr = (ensemble['metrics'].get('best_sl_atr', 2.0), ensemble['metrics'].get('best_tp_atr', 4.0)) if ensemble else (2.0, 4.0)

        if not ml_preds.empty and ensemble and ensemble['ensemble_signal'] is not None:
            # Define a search space for the ML confidence threshold
            threshold_options = np.arange(0.5, 0.7, 0.02).round(2) # e.g., 0.50, 0.52, ..., 0.68
            
            original_ens_sig = ensemble['ensemble_signal'].copy()

            for threshold in threshold_options:
                current_filtered_signal = original_ens_sig.copy()
                ml_preds_aligned = ml_preds.reindex(current_filtered_signal.index, fill_value=0)
                
                # Filter signals based on ML confidence
                # If ensemble signal is long (1) and ML predicts 0 (down), set to 0
                # If ensemble signal is short (-1) and ML predicts 1 (up), set to 0
                current_filtered_signal[ (current_filtered_signal == 1) & (ml_preds_aligned < threshold) ] = 0
                current_filtered_signal[ (current_filtered_signal == -1) & (ml_preds_aligned > (1-threshold)) ] = 0 # If ML is confident it's UP, filter SHORT signals

                pnl_filtered, fq_filtered = simulate_trades(df, current_filtered_signal, sl_ens_atr, tp_ens_atr, cfg['point_value'], cfg['commission'], cfg['slippage_points'], asset_name=symbol)
                metrics_filtered = metrics_from_pnl(pnl_filtered)
                score_filtered = penalized_score(metrics_filtered, fq_filtered, 0)

                if score_filtered > best_ml_filtered_score:
                    best_ml_filtered_score = score_filtered
                    best_ml_threshold = threshold
            
            print(f"  → Optimized ML confidence threshold: {best_ml_threshold:.2f} with score {best_ml_filtered_score:.1f}")

        # --- IMPROVEMENT: Create and evaluate the final ML-filtered signal using the optimized threshold ---
        final_signal = ens_sig.copy()
        ml_preds_aligned = ml_preds.reindex(final_signal.index, fill_value=0)
        final_signal[ (final_signal == 1) & (ml_preds_aligned < best_ml_threshold) ] = 0
        final_signal[ (final_signal == -1) & (ml_preds_aligned > (1-best_ml_threshold)) ] = 0
        final_pnl, _ = simulate_trades(df, final_signal, sl_ens_atr, tp_ens_atr, cfg['point_value'], cfg['commission'], cfg['slippage_points'], asset_name=symbol)
        final_metrics = metrics_from_pnl(final_pnl)

    out = {
        'symbol': symbol,
        'top_strategies': [ {'family':i['family'],'params':i['params'],'sl_atr':i['sl_atr'],'tp_atr':i['tp_atr']} for i in (top_inds or []) ],
        'ensemble_metrics': ensemble['metrics'] if ensemble else None,
        'final_ml_filtered_metrics': final_metrics,
        'ml_aucs': aucs,
        'data_points': len(df)
    }
    # Plot the final equity curve
    if final_pnl:
        eq = np.cumsum(final_pnl) + CAPITAL
        plt.figure(figsize=(10,4)); plt.plot(eq); plt.title(f"{symbol} Final ML-Filtered Equity"); plt.grid(True)
        plt.savefig(os.path.join(OUT_PATH, f"{symbol}_final_equity.png"), dpi=200); plt.close()
    return out

# ---------------------------
# RUN ALL SYMBOLS
# ---------------------------
if __name__ == "__main__":
    all_out = {}
    for sym, patterns in SYMBOL_FILES.items():
        try:
            res = run_for_symbol(sym, patterns)
            all_out[sym] = res
        except Exception as e:
            print(f"[ERRO] ao processar {sym}: {e}")
    with open(os.path.join(OUT_PATH, "pipeline_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2, ensure_ascii=False)
    print("\n==== PIPELINE FINALIZADO ====")

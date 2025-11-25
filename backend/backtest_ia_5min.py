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
from datetime import timedelta
from collections import defaultdict
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

# ML
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# ---------------------------
# CONFIG
# ---------------------------
BASE_PATH = r"C:\Users\Marco\Documents\dados"
OUT_PATH = r"C:\Users\Marco\ia_finance"
os.makedirs(OUT_PATH, exist_ok=True)

# If you want to specify explicit filenames, put them in the lists.
# You may also supply globs like "WIN*.csv" or "WIN$*.csv" and the script will expand them.
SYMBOL_FILES = {
    "PETR4": ["PETR4.csv"],
    "WIN":   ["WIN*.csv", "WIN$*.csv"],
    "WDO":   ["WDO*.csv", "WDO$*.csv"]
}

CAPITAL = 100_000
WF_TRAIN_MONTHS = 6
WF_TEST_MONTHS = 2
LOOKAHEAD_BARS = 20  # intrabar check
MIN_TRADES = 30      # regras que façam < MIN_TRADES são desconsideradas
N_GEN = 30           # gerações do otimizador genético (reduzi para velocidade; aumente se quiser)
POP_SIZE = 30
ELITE = 6
TOP_K_ENSEMBLE = 6
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# parallel jobs
CPU_COUNT = max(1, multiprocessing.cpu_count() - 1)

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
            glob_path = os.path.join(BASE_PATH, pat)
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

def load_symbol_continuous(patterns_or_files):
    files = expand_file_list(patterns_or_files)
    if not files:
        print(f"[WARN] nenhum arquivo encontrado para padrões: {patterns_or_files}")
        return None
    dfs = []
    for f in files:
        try:
            df = load_mt5_csv(f)
            dfs.append(df)
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
def maybe_to_m5(df):
    dif = df.index.to_series().diff().dropna()
    if len(dif) == 0:
        return df
    med_minutes = int(pd.Timedelta(dif.median()).seconds // 60)
    # if median diff is 1 minute, convert to 5min
    if med_minutes == 1:
        df5 = df.resample("5T").agg({
            'open':'first','high':'max','low':'min','close':'last',
            'tickvol': 'sum' if 'tickvol' in df.columns else 'last',
            'vol': 'sum' if 'vol' in df.columns else 'last',
            'spread': 'mean' if 'spread' in df.columns else 'last'
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

    # spans usados pelo GA (inclui 5,8,9,13,21,34,50,89,100,200)
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

    df = df.dropna()
    return df

# ---------------------------
# Simulação intrabar (com tratamento de TP/SL e ordem de ocorrência)
# ---------------------------
def simulate_trades(df, signals, sl_points, tp_points, point_value, size=1, lookahead=LOOKAHEAD_BARS):
    pnl = []
    false_quick_stops = 0
    # ensure signals index aligns
    signals = signals.reindex(df.index, fill_value=False)
    for i in range(len(df) - lookahead - 2):
        if not signals.iloc[i]:
            continue
        entrada = df['open'].iloc[i+1]
        highs = df['high'].iloc[i+1:i+1+lookahead].values
        lows = df['low'].iloc[i+1:i+1+lookahead].values

        hit_tp = np.where(highs >= entrada + tp_points)[0]
        hit_sl = np.where(lows <= entrada - sl_points)[0]

        pnl_pts = None
        if hit_tp.size > 0 and hit_sl.size > 0:
            tp_idx = int(hit_tp[0]); sl_idx = int(hit_sl[0])
            pnl_pts = tp_points if tp_idx < sl_idx else -sl_points
            if (pnl_pts < 0) and sl_idx <= 3:
                false_quick_stops += 1
        elif hit_tp.size > 0:
            pnl_pts = tp_points
        elif hit_sl.size > 0:
            pnl_pts = -sl_points
            if 0 <= int(hit_sl[0]) <= 3:
                false_quick_stops += 1
        else:
            fechamento = df['close'].iloc[i+1+lookahead-1]
            pnl_pts = (fechamento - entrada)

        pnl_money = pnl_pts * point_value * size
        pnl.append(pnl_money)
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
    sharpe = float((arr.mean() / (arr.std() or 1.0)) * math.sqrt(252))
    return {
        "profit": profit,
        "trades": int(len(arr)),
        "winrate": winrate,
        "pf": round(pf,3),
        "maxdd": round(maxdd,2),
        "sharpe": round(sharpe,3)
    }

def penalized_score(metrics, false_quick_stops, months_no_trade):
    if not metrics:
        return -1e9
    score = metrics['profit']*0.6 + metrics['pf']*200 + metrics['sharpe']*50 - metrics['maxdd']*2 + metrics['winrate']*0.5
    if metrics['trades'] < MIN_TRADES:
        score -= 5000 + (MIN_TRADES - metrics['trades'])*100
    score -= false_quick_stops * 50
    score -= months_no_trade * 200
    return score

# ---------------------------
# Strategy families generation (rule-based)
# ---------------------------
def generate_rule_population(seed_pop=None):
    pop = []
    families = ['pullback','macross','volbreak']
    if seed_pop:
        pop.extend(seed_pop)
    while len(pop) < POP_SIZE:
        f = random.choice(families)
        if f == 'pullback':
            p = {'vol_mult': random.choice([1.1,1.3,1.5,1.8,2.0])}
            sl,tp = random.choice([ (300,600),(400,800),(500,1000)])
        elif f == 'macross':
            fast = random.choice([5,8,9,13,21])
            slow = random.choice([21,34,50,89])
            if fast >= slow: fast = max(5, slow-5)
            p = {'fast':fast,'slow':slow}
            sl,tp = random.choice([(300,600),(400,800),(600,1200)])
        else:
            lookback = random.choice([10,15,20,30])
            mult = random.choice([1.1,1.3,1.5,1.8])
            p = {'lookback':lookback,'mult':mult}
            sl,tp = random.choice([(300,600),(400,800),(500,1000)])
        pop.append({'family':f,'params':p,'sl':sl,'tp':tp})
    return pop

# ---------------------------
# Strategy signals (com defensivas)
# ---------------------------
def ensure_ema_exists(df, span):
    col = f'ema_{span}'
    if col not in df.columns:
        df[col] = df['close'].ewm(span=span, adjust=False).mean()

def signal_from_strategy(df, ind):
    fam = ind['family']; p = ind['params']
    # defensive: ensure EMAs needed exist
    if fam == 'macross':
        for s in [p.get('fast'), p.get('slow')]:
            if s is not None:
                ensure_ema_exists(df, s)

    # compute signals
    if fam == 'pullback':
        # defensive ensure
        for s in [21,50,200]:
            ensure_ema_exists(df, s)
        signal = (df['close'] < df['ema_21']) & (df['close'].shift(1) >= df['ema_21'].shift(1))
        signal &= (df['vol'] > df['vol_med'] * p['vol_mult'])
        signal &= (df['close'] > df['ema_50']) & (df['ema_50'] > df['ema_200'])
        signal &= df.index.hour.to_series().between(10,17).values
    elif fam == 'macross':
        fast = p['fast']; slow = p['slow']
        signal = (df[f'ema_{fast}'] > df[f'ema_{slow}']) & (df[f'ema_{fast}'].shift(1) <= df[f'ema_{slow}'].shift(1))
    elif fam == 'volbreak':
        lookback = p['lookback']; mult = p['mult']
        rolling_std = df['close'].pct_change().rolling(lookback).std()
        threshold = rolling_std * mult
        candle_move = (df['close'] - df['open']).abs() / df['open'].shift(1)
        signal = (candle_move > threshold) & (df['vol'] > df['vol_med']*0.8)
        signal &= df.index.hour.to_series().between(10,17).values
    else:
        signal = pd.Series(False, index=df.index)
    return signal.astype(bool)

# ---------------------------
# Walk-forward evaluation
# ---------------------------
def months_between(a,b):
    return (b.year - a.year)*12 + (b.month - a.month)

def walkforward_evaluate(df_full, indiv, asset_name):
    start = df_full.index.min()
    end = df_full.index.max()
    cur = start
    folds = []
    months_no_trade = 0
    while True:
        train_end = (cur + pd.DateOffset(months=WF_TRAIN_MONTHS)) - pd.Timedelta(seconds=1)
        test_end = train_end + pd.DateOffset(months=WF_TEST_MONTHS)
        if test_end > end:
            break
        train = df_full[cur:train_end]
        test = df_full[train_end + pd.Timedelta(seconds=1):test_end]
        cur = cur + pd.DateOffset(months=WF_TEST_MONTHS)
        if len(train) < 50 or len(test) < 10:
            continue
        sig = signal_from_strategy(test, indiv)
        # choose sl/tp based on asset type
        if 'WIN' in asset_name:
            point_val = 5
        elif 'WDO' in asset_name:
            point_val = 0.05
        else:
            point_val = 0.01
        pnl, false_quick = simulate_trades(test, sig, indiv['sl'], indiv['tp'], point_val)
        mets = metrics_from_pnl(pnl)
        if mets is None or mets['trades'] == 0:
            months_no_trade += months_between(test.index.min(), test.index.max())
            continue
        mets['false_quick'] = false_quick
        folds.append(mets)
    if not folds:
        return None
    agg = {
        'profit_mean': float(np.mean([f['profit'] for f in folds])),
        'profit_sum': float(np.sum([f['profit'] for f in folds])),
        'sharpe_mean': float(np.mean([f['sharpe'] for f in folds])),
        'dd_mean': float(np.mean([f['maxdd'] for f in folds])),
        'pf_mean': float(np.mean([f['pf'] for f in folds])),
        'winrate_mean': float(np.mean([f['winrate'] for f in folds])),
        'trades_sum': int(np.sum([f['trades'] for f in folds])),
        'folds': len(folds),
        'months_no_trade': months_no_trade
    }
    agg['score'] = penalized_score({'profit':agg['profit_mean'],'pf':agg['pf_mean'],'sharpe':agg['sharpe_mean'],'maxdd':agg['dd_mean'],'winrate':agg['winrate_mean'],'trades':agg['trades_sum']}, 0, agg['months_no_trade'])
    return agg

# ---------------------------
# Genetic-like optimization
# ---------------------------
def evolve_population(df, asset_name):
    pop = generate_rule_population()
    evaluated = []
    for gen in range(N_GEN):
        results = Parallel(n_jobs=CPU_COUNT)(delayed(walkforward_evaluate)(df, ind, asset_name) for ind in pop)
        scored = []
        for ind, res in zip(pop, results):
            sc = res['score'] if res else -1e9
            scored.append((sc, ind, res))
        scored.sort(key=lambda x: x[0], reverse=True)
        elite = [x[1] for x in scored[:ELITE] if x[1] is not None]
        evaluated.extend(scored[:ELITE])
        best = scored[0]
        best_score = best[0]
        best_ind = best[1]
        best_res = best[2]
        print(f"[EVO {asset_name}] GEN {gen+1}/{N_GEN} best score {best_score:.1f} family {best_ind['family']} params {best_ind['params']} trades {best_res['trades_sum'] if best_res else 0}")
        newpop = elite.copy()
        # if elite smaller than needed, refill with random
        while len(newpop) < POP_SIZE:
            if len(elite) >= 2:
                p1 = random.choice(elite); p2 = random.choice(elite)
                child = crossover(p1, p2)
            else:
                child = random.choice(generate_rule_population())
            child = mutate(child)
            newpop.append(child)
        pop = newpop
    evaluated_sorted = sorted(evaluated, key=lambda x: x[0], reverse=True)
    top = []
    for score, ind, res in evaluated_sorted:
        if len(top) >= TOP_K_ENSEMBLE:
            break
        if ind is None: 
            continue
        key = (ind['family'], json.dumps(ind['params'], sort_keys=True), ind['sl'], ind['tp'])
        if key in [ (t['family'], json.dumps(t['params'], sort_keys=True), t['sl'], t['tp']) for t in top ]:
            continue
        top.append(ind)
    return top

def crossover(a,b):
    child = {'family': a['family'] if random.random()<0.5 else b['family'], 'params':{}, 'sl':a['sl'], 'tp':a['tp']}
    for k in set(list(a['params'].keys()) + list(b['params'].keys())):
        child['params'][k] = a['params'].get(k, b['params'].get(k))
        if random.random() < 0.5:
            child['params'][k] = b['params'].get(k, a['params'].get(k))
    child['sl'] = int((a['sl'] + b['sl'])/2) if random.random()<0.7 else (a['sl'] if random.random() < 0.5 else b['sl'])
    child['tp'] = int((a['tp'] + b['tp'])/2) if random.random()<0.7 else (a['tp'] if random.random() < 0.5 else b['tp'])
    return child

def mutate(ind):
    if random.random() < 0.2:
        if ind['family']=='pullback':
            ind['params']['vol_mult'] = max(1.05, ind['params']['vol_mult'] + random.choice([-0.2,-0.1,0.1,0.2]))
        elif ind['family']=='macross':
            ind['params']['fast'] = max(3, ind['params']['fast'] + random.choice([-2,-1,1,2]))
            ind['params']['slow'] = max(ind['params']['fast']+1, ind['params']['slow'] + random.choice([-3,-1,1,3]))
        else:
            ind['params']['lookback'] = max(5, ind['params']['lookback'] + random.choice([-5,-2,2,5]))
            ind['params']['mult'] = max(0.8, ind['params']['mult'] + random.choice([-0.2,-0.1,0.1,0.2]))
    if random.random() < 0.15:
        ind['sl'] = max(1, int(ind['sl'] * random.choice([0.9,1.0,1.1,1.2])))
        ind['tp'] = max(1, int(ind['tp'] * random.choice([0.9,1.0,1.1,1.2])))
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
        if 'WIN' in asset_name:
            point = 5
        elif 'WDO' in asset_name:
            point = 0.05
        else:
            point = 0.01
        pnl, false_quick = simulate_trades(df, sig, ind['sl'], ind['tp'], point)
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
    ensemble_signal = pd.Series(weighted > 0.5, index=df.index)
    # ensemble pnl
    if 'WIN' in asset_name:
        point = 5
    elif 'WDO' in asset_name:
        point = 0.05
    else:
        point = 0.01
    mean_sl = int(np.mean([i['ind']['sl'] for i in infos]))
    mean_tp = int(np.mean([i['ind']['tp'] for i in infos]))
    pnl_ens, fq = simulate_trades(df, ensemble_signal, mean_sl, mean_tp, point)
    mets = metrics_from_pnl(pnl_ens)
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

def train_lgbm_walkforward(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    aucs = []
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        dtrain = lgb.Dataset(Xtr, ytr)
        params = {
            'objective':'binary',
            'metric':'auc',
            'verbosity':-1,
            'boosting_type':'gbdt',
            'seed':SEED,
            'num_threads':2
        }
        bst = lgb.train(params, dtrain, num_boost_round=200, valid_sets=[dtrain], verbose_eval=False)
        ypred = bst.predict(Xte)
        auc = roc_auc_score(yte, ypred) if len(np.unique(yte))>1 else 0.5
        models.append(bst); aucs.append(float(auc))
    return models, aucs

# ---------------------------
# Main pipeline per symbol
# ---------------------------
def run_for_symbol(symbol, patterns_or_files):
    print(f"\n===== PROCESSANDO {symbol} =====")
    df_raw = load_symbol_continuous(patterns_or_files)
    if df_raw is None or len(df_raw) < 200:
        print(f"[ERRO] dados insuficientes para {symbol} (len={0 if df_raw is None else len(df_raw)})")
        return None
    df = maybe_to_m5(df_raw)
    df = add_features(df)
    top_inds = evolve_population(df, symbol)
    ensemble = build_ensemble(df, top_inds, symbol)
    ens_sig = ensemble['ensemble_signal'] if ensemble else pd.Series(False, index=df.index)
    X, y, df_ml = prepare_ml_dataset(df, ens_sig, horizon=6)
    if len(X) < 200:
        print("[WARN] Dados insuficientes para treinar ML")
        models = []; aucs = []
    else:
        models, aucs = train_lgbm_walkforward(X,y)
    out = {
        'symbol': symbol,
        'top_strategies': [ {'family':i['family'],'params':i['params'],'sl':i['sl'],'tp':i['tp']} for i in (top_inds or []) ],
        'ensemble_metrics': ensemble['metrics'] if ensemble else None,
        'ml_aucs': aucs,
        'data_points': len(df)
    }
    if ensemble and ensemble['pnl']:
        eq = np.cumsum(ensemble['pnl']) + CAPITAL
        plt.figure(figsize=(10,4))
        plt.plot(eq); plt.title(f"{symbol} Ensemble equity"); plt.grid(True)
        plt.savefig(os.path.join(OUT_PATH, f"{symbol}_ensemble_equity.png"), dpi=200)
        plt.close()
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

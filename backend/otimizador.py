# otimizador.py — OTIMIZAÇÃO 100% FUNCIONAL
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools
import os
from sinais import calculate_rsi, calculate_macd, calculate_bollinger

if not mt5.initialize():
    print("MT5 não conectado")
    exit()

ATIVO = "WIN$"
TIMEFRAME = mt5.TIMEFRAME_M5
DIAS = 30
PASTA = "C:/temp/otimizacao"
os.makedirs(PASTA, exist_ok=True)

# === PARÂMETROS ===
param_grid = {
    'confianca_min': [0.4, 0.5, 0.6],
    'rsi_min': [20, 25, 30],
    'rsi_max': [70, 75, 80],
    'vol_min': [0.0003, 0.0005, 0.0008],
    'macd_fast': [8, 10, 12],
    'macd_slow': [17, 20, 26]
}

# === DADOS ===
def get_data():
    ate = datetime.now()
    de = ate - timedelta(days=DIAS)
    while de.weekday() >= 5: de -= timedelta(days=1)
    while ate.weekday() >= 5: ate -= timedelta(days=1)
    rates = mt5.copy_rates_range(ATIVO, TIMEFRAME, de, ate)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

df = get_data()
print(f"{len(df)} velas carregadas")

# === BACKTEST RÁPIDO ===
def backtest_config(params):
    sinais = []
    ultimo_sinal = None
    i = 200

    while i < len(df):
        janela = df.iloc[i-200:i].copy()
        close = janela['close']
        volume = janela['tick_volume']

        # Features
        janela['retorno'] = close.pct_change()
        janela['rsi'] = calculate_rsi(close)
        macd, macd_sig = calculate_macd(close, params['macd_fast'], params['macd_slow'])
        janela['macd'] = macd
        janela['macd_signal'] = macd_sig
        janela['volume_sma'] = volume.rolling(20).mean()

        ultimo = janela.iloc[-1]
        anterior = janela.iloc[-2]

        # Cruzamentos
        compra = (ultimo['macd'] > ultimo['macd_signal']) and (anterior['macd'] <= anterior['macd_signal'])
        venda = (ultimo['macd'] < ultimo['macd_signal']) and (anterior['macd'] >= anterior['macd_signal'])

        # Filtros
        if (ultimo['rsi'] < params['rsi_min'] or 
            ultimo['rsi'] > params['rsi_max'] or
            janela['retorno'].tail(20).std() < params['vol_min']):
            i += 1
            continue

        confianca = abs(ultimo['macd'] - ultimo['macd_signal']) / (abs(ultimo['macd_signal']) + 1e-8)
        if confianca < params['confianca_min']:
            i += 1
            continue

        sinal = None
        if compra and ultimo_sinal != "COMPRA":
            sinal = "COMPRA"
            ultimo_sinal = "COMPRA"
        elif venda and ultimo_sinal != "VENDA":
            sinal = "VENDA"
            ultimo_sinal = "VENDA"

        if sinal:
            sinais.append({
                'time': df.index[i],
                'sinal': sinal,
                'preco': round(close.iloc[-1], 0)
            })

        i += 1

    if len(sinais) < 2:
        return {'sinais': len(sinais), 'pnl': 0, 'winrate': 0, 'params': params}

    # === CÁLCULO DE PnL (CORRIGIDO) ===
    df_sinais = pd.DataFrame(sinais)
    pnl_total = 0
    acertos = 0

    for i in range(len(df_sinais)):
        entrada = df_sinais.iloc[i]
        # Próximo sinal oposto
        saida_preco = None
        for j in range(i+1, len(df_sinais)):
            if df_sinais.iloc[j]['sinal'] != entrada['sinal']:
                saida_preco = df_sinais.iloc[j]['preco']
                break
        if saida_preco is None:
            saida_preco = df['close'].iloc[-1]  # Última cotação

        pontos = (saida_preco - entrada['preco']) if entrada['sinal'] == 'COMPRA' else (entrada['preco'] - saida_preco)
        pnl = pontos * 0.20
        pnl_total += pnl
        if pnl > 0:
            acertos += 1

    winrate = round(acertos / len(df_sinais) * 100, 1) if len(df_sinais) > 0 else 0

    return {
        'sinais': len(df_sinais),
        'pnl': round(pnl_total, 2),
        'winrate': winrate,
        'params': params
    }

# === OTIMIZAÇÃO ===
resultados = []
combos = list(itertools.product(
    param_grid['confianca_min'], param_grid['rsi_min'], param_grid['rsi_max'],
    param_grid['vol_min'], param_grid['macd_fast'], param_grid['macd_slow']
))

print(f"Testando {len(combos)} combinações...")

for idx, combo in enumerate(combos):
    if idx % 50 == 0:
        print(f"Progresso: {idx}/{len(combos)}")
    
    params = {
        'confianca_min': combo[0],
        'rsi_min': combo[1],
        'rsi_max': combo[2],
        'vol_min': combo[3],
        'macd_fast': combo[4],
        'macd_slow': combo[5]
    }
    res = backtest_config(params)
    resultados.append(res)

# === MELHOR CONFIG ===
melhor = max(resultados, key=lambda x: x['pnl'] + x['sinais'] * 10)
print("\n" + "="*60)
print("MELHOR CONFIGURAÇÃO ENCONTRADA")
print("="*60)
print(f"Confiança mínima: {melhor['params']['confianca_min']*100:.0f}%")
print(f"RSI: {melhor['params']['rsi_min']} – {melhor['params']['rsi_max']}")
print(f"Volatilidade mínima: {melhor['params']['vol_min']}")
print(f"MACD: ({melhor['params']['macd_fast']}, {melhor['params']['macd_slow']})")
print(f"Sinais: {melhor['sinais']}")
print(f"Winrate: {melhor['winrate']}%")
print(f"PnL Total: R$ {melhor['pnl']:,}")
print("="*60)

# === SALVAR ===
pd.DataFrame(resultados).to_excel(f"{PASTA}/otimizacao_completa.xlsx", index=False)
print(f"\nRelatório salvo em: {PASTA}/otimizacao_completa.xlsx")
# ia_5min_mt5.py — DASHBOARD + SAÍDAS PARCIAIS (1.25 / 1.50 / 2.00)
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import winsound
from sqlalchemy import create_engine
from backend.sinais import initialize_mt5, selecionar_simbolo, gerar_sinal_IA, executar_ordem

# === CONFIGURAÇÃO ===
st.set_page_config(page_title="IA 5min - SAÍDAS PARCIAIS", layout="wide")
engine = create_engine('postgresql://postgres:admin12345@localhost:5432/ia_mt5')

if not initialize_mt5():
    st.error("ERRO: MT5 não conectado. Abra o MT5 e configure uma conta válida.")
    st.stop()

# === ATIVOS COM SAÍDAS PARCIAIS ===
ATIVOS = {
    "IBOV": {
        "symbol": "WINZ25", "nome": "IBOV (via WIN)", "decimais": 0,
        "ponto": 5, "sl_pontos": 100, "tp_pontos": [125, 150, 200]
    },
    "WDO": {
        "symbol": "WDOX25", "nome": "Mini-Dólar", "decimais": 1,
        "ponto": 0.05, "sl_pontos": 20, "tp_pontos": [25, 30, 40]
    },
    "PETR4": {
        "symbol": "PETR4", "nome": "Petrobras", "decimais": 2,
        "ponto": 0.01, "sl_pontos": 50, "tp_pontos": [62.5, 75, 100]
    }
}

# Estado
if 'ultimo_sinal' not in st.session_state:
    st.session_state.ultimo_sinal = {}
if 'ordens' not in st.session_state:
    st.session_state.ordens = []

# === FUNÇÕES ===
def tocar_som(tipo="COMPRA"):
    freq = 1200 if tipo == "COMPRA" else 600
    try:
        winsound.Beep(freq, 800)
    except:
        pass

def calcular_sl_tp(preco, sinal, config):
    ponto = config["ponto"]
    sl_pontos = config["sl_pontos"]
    tp_pontos = config["tp_pontos"]

    sl = preco - (sl_pontos * ponto) if sinal == "COMPRA" else preco + (sl_pontos * ponto)
    sl = round(sl, config["decimais"])

    tps = []
    for tp_p in tp_pontos:
        if sinal == "COMPRA":
            tps.append(round(preco + tp_p * ponto, config["decimais"]))
        else:
            tps.append(round(preco - tp_p * ponto, config["decimais"]))
    return sl, tps

# === AQUI ESTÁ A ÚNICA PARTE REALMENTE CORRIGIDA ===
def enviar_ordem_parcial(ativo, sinal, preco, config):
    sl, tps = calcular_sl_tp(preco, sinal, config)

    # >>> CORREÇÃO MÍNIMA: volume correto para cada ativo <<<
    volume_parcial = 1.0 if config["decimais"] == 0 else 0.01

    resultados = []

    for i, tp in enumerate(tps, 1):
        sucesso = executar_ordem(ativo, sinal, preco, sl, tp, volume_parcial, i)

        # >>> CORREÇÃO MÍNIMA: evitar travamento do MT5 <<<
        time.sleep(1)

        if sucesso:
            resultados.append(f"TP{i} ({tp}) → OK")
            st.session_state.ordens.append({
                'time': time.strftime("%H:%M:%S"),
                'ativo': config["nome"],
                'sinal': sinal,
                'preco': preco,
                'sl': sl,
                'tp': tp,
                'volume': volume_parcial,
                'status': 'EXECUTADA'
            })
        else:
            resultados.append(f"TP{i} → Erro")

    msg = f"{sinal} {3*volume_parcial:.2f} contratos (3 saídas):\n" + "\n".join(resultados)
    return all("OK" in r for r in resultados), msg

# === DASHBOARD ===
st.title("IA 5min - SAÍDAS PARCIAIS (1.25 / 1.50 / 2.00)")
st.markdown("**1 Clique → 3 Ordens com TP em 1.25x, 1.50x, 2.00x**")

count = st_autorefresh(interval=5000, key="refresh")
hora = time.strftime("%H:%M")
aberto = 9 <= int(hora[:2]) <= 18
st.write(f"**Atualização #{count}** | {hora} | Mercado: **{'ABERTO' if aberto else 'FECHADO'}**")

col1, col2, col3 = st.columns(3)

for ativo, config in ATIVOS.items():
    if not selecionar_simbolo(config["symbol"]):
        with (col1 if ativo == "IBOV" else col2 if ativo == "WDO" else col3):
            st.warning(f"{config['nome']} indisponível")
        continue

    sinal, preco = gerar_sinal_IA(config["symbol"], mt5.TIMEFRAME_M5)
    preco = round(preco, config["decimais"]) if preco else 0

    if sinal and sinal != st.session_state.ultimo_sinal.get(ativo):
        st.session_state.ultimo_sinal[ativo] = sinal
        st.balloons()
        tocar_som(sinal)

    with (col1 if ativo == "IBOV" else col2 if ativo == "WDO" else col3):
        st.subheader(f"{config['nome']} ({config['symbol']})")

        if ativo == "IBOV":
            st.metric("WIN", f"{preco:,}")
            st.metric("IBOV", f"{(preco*5):,} pts")
        else:
            st.metric("Preço", f"{preco}")

        if sinal:
            cor = "green" if sinal == "COMPRA" else "red"
            st.markdown(f"<h2 style='color:{cor};'>→ {sinal}</h2>", unsafe_allow_html=True)

            sl, tps = calcular_sl_tp(preco, sinal, config)
            st.caption(f"SL: {sl} | TP1: {tps[0]} | TP2: {tps[1]} | TP3: {tps[2]}")

            if st.button(f"EXECUTAR {sinal} (3 SAÍDAS)", key=f"btn_{ativo}", type="primary"):
                sucesso, msg = enviar_ordem_parcial(config["symbol"], sinal, preco, config)
                if sucesso:
                    st.success(msg)
                else:
                    st.error(msg)
        else:
            st.info("Aguardando sinal...")

# === ORDENS EXECUTADAS ===
if st.session_state.ordens:
    st.subheader("Ordens com Saídas Parciais")
    df_ordens = pd.DataFrame(st.session_state.ordens)
    st.dataframe(df_ordens, use_container_width=True)

st.markdown("---")
st.caption("IA 5min Elite | Saídas em 1.25x / 1.50x / 2.00x | R:R Otimizado")
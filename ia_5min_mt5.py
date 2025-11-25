# ia_5min_mt5.py — STREAMLIT DASHBOARD (usa backend/sinais.py)
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import MetaTrader5 as mt5
import pandas as pd
import time
import winsound
from backend import sinais

# === CONFIG ===
st.set_page_config(page_title="IA 5min - Saídas Parciais (lock)", layout="wide")
ATIVOS = {
    "IBOV": {"symbol": "WINZ25", "nome": "IBOV (via WIN)", "ponto": 5, "decimais": 0, "sl_pontos": 100, "tp_pontos":[125,150,200]},
    "WDO":  {"symbol": "WDOX25", "nome":"Mini-Dólar", "ponto":0.05, "decimais":1, "sl_pontos":20, "tp_pontos":[25,30,40]},
    "PETR4":{"symbol":"PETR4", "nome":"Petrobras", "ponto":0.01, "decimais":2, "sl_pontos":50, "tp_pontos":[62.5,75,100]}
}

# inicializa MT5 (tenta)
if not sinais.initialize_mt5():
    st.warning("MT5 não conectado — verifique terminal.")
    # não st.stop() porque você pode querer usar CSV/hist

# Session state
if 'ultimo_sinal' not in st.session_state:
    st.session_state.ultimo_sinal = {}

if 'ordens' not in st.session_state:
    st.session_state.ordens = []

# autorefresh a cada 10s (ajuste ao seu gosto)
count = st_autorefresh(interval=10_000, key="refresh")

st.title("IA 5min - Saídas Parciais (lock)")
st.write(f"Atualização #{count} | {time.strftime('%H:%M:%S')} | Lock: {'ON' if sinais.sinal_bloqueado() else 'OFF'}")

cols = st.columns(3)

for i,(k, cfg) in enumerate(ATIVOS.items()):
    col = cols[i]
    with col:
        st.subheader(f"{cfg['nome']} ({cfg['symbol']})")
        # seleciona símbolo no MT5 (safe)
        disponível = sinais.selecionar_simbolo(cfg['symbol'])
        if not disponível:
            st.warning("Símbolo indisponível no MT5")
            continue

        # --- IMPROVEMENT: Unpack all values from the backend ---
        sinal, preco, confianca, sl_atr, tp_atr, atr_val = sinais.gerar_sinal_IA(cfg['symbol'], mt5.TIMEFRAME_M5)
        preco_display = round(preco, cfg['decimais']) if preco else None

        if sinal:
            color = "green" if sinal == "COMPRA" else "red"
            st.markdown(f"<h2 style='color:{color};'>→ {sinal} @ {preco_display}</h2>", unsafe_allow_html=True)
            
            # --- IMPROVEMENT: Display dynamic SL/TP and confidence ---
            sl_em_reais = (sl_atr * atr_val) if atr_val else (cfg['sl_pontos'] * cfg['ponto'])
            tp_em_reais = (tp_atr * atr_val) if atr_val else (cfg['tp_pontos'][0] * cfg['ponto'])
            st.caption(f"Confiança: {confianca:.2%} | SL ATR: {sl_atr} | TP ATR: {tp_atr}")

            if st.button(f"EXECUTAR {sinal} (3 SAÍDAS) - {cfg['symbol']}", key=f"btn_{cfg['symbol']}"):
                # --- IMPROVEMENT: Use dynamic SL/TP for order execution ---
                sl = preco - sl_em_reais if sinal == "COMPRA" else preco + sl_em_reais
                tps = []
                # Use the optimized TP multiplier for the first exit, then scale for subsequent exits
                for tp_mul in cfg['tp_pontos']:
                    # This is a simple scaling example. You might want a more sophisticated logic.
                    current_tp_atr = tp_atr * (tp_mul / cfg['tp_pontos'][0])
                    if sinal == "COMPRA":
                        tps.append(preco + (current_tp_atr * atr_val))
                    else:
                        tps.append(preco - (current_tp_atr * atr_val))
                # envia TPs parciais (volume 1 por TP)
                resultados = []
                sucesso_total = True
                for idx,tp in enumerate(tps, start=1):
                    ok = sinais.executar_ordem(cfg['symbol'], sinal, preco, sl, tp, 1.0, idx)
                    resultados.append(f"TP{idx} ({round(tp,cfg['decimais'])}) → {'OK' if ok else 'Erro'}")
                    if not ok:
                        sucesso_total = False
                # gravar estado local de ordens para exibição
                st.session_state.ordens.append({
                    "time": time.strftime("%H:%M:%S"),
                    "ativo": cfg['symbol'],
                    "sinal": sinal,
                    "preco": round(preco_display, cfg['decimais']) if preco_display else preco_display,
                    "resultados": resultados
                })
                # bloquear sinais (já feito internamente por salvar_sinal_db quando gerou)
                # tocar som
                try:
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                except Exception:
                    pass
                if sucesso_total:
                    st.success("\n".join(resultados))
                else:
                    st.error("\n".join(resultados))
        else:
            st.info("Aguardando sinal...")

# exibir ordens realizadas
if st.session_state.ordens:
    st.subheader("Ordens recentes")
    st.dataframe(pd.DataFrame(st.session_state.ordens).iloc[::-1], use_container_width=True)

# chamada periódica para liberar lock se ordens fechadas
if sinais.sinal_bloqueado():
    fechado = sinais.verificar_ordens_fechadas()
    if fechado and not sinais.sinal_bloqueado():
        st.session_state.force_refresh = True
        st.rerun()

# se o backend liberou o lock, forçar refresh
if st.session_state.get("force_refresh", False):
    st.session_state.force_refresh = False
    st.rerun()

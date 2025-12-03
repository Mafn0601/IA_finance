import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import MetaTrader5 as mt5
import winsound

# altair para mini-charts
import altair as alt

from backend import sinais

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="IA Finance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== THEME / SESSION STATE ======
if 'theme' not in st.session_state:
    nowh = datetime.now().hour
    st.session_state.theme = 'dark' if nowh >= 18 or nowh < 6 else 'light'

def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

# ====== CSS + STYLES (Glass UI + switch + animations) ======
PRIMARY_BG = "#071029" if st.session_state.theme == 'dark' else "#f7fafc"
APP_TEXT = "#e6eef6" if st.session_state.theme == 'dark' else "#0f172a"
CARD_BG = "rgba(10,18,28,0.7)" if st.session_state.theme == 'dark' else "rgba(255,255,255,0.6)"
GLASS_BORDER = "rgba(255,255,255,0.04)" if st.session_state.theme == 'dark' else "rgba(15,23,42,0.06)"

css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
  background: {PRIMARY_BG} !important;
  color: {APP_TEXT} !important;
  font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}}

/* Sidebar glass */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(8,16,25,0.75), rgba(10,18,28,0.75));
  border-right: 1px solid {GLASS_BORDER};
  backdrop-filter: blur(6px);
}}
/* Main container padding */
.block-container {{ padding: 1.2rem 2rem 2rem 2rem !important; }}

/* Header */
h1 {{ font-weight:800; margin-bottom: 4px; }}

/* Card */
.card {{
  background: {CARD_BG};
  border-radius: 14px;
  padding: 18px;
  border: 1px solid {GLASS_BORDER};
  box-shadow: 0 6px 30px rgba(0,0,0,0.4);
  transition: transform 0.18s ease, box-shadow 0.18s ease;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}}
.card:hover {{
  transform: translateY(-6px);
  box-shadow: 0 18px 50px rgba(2,6,23,0.6);
}}

/* Title / price */
.asset-title {{
  font-size: 20px; font-weight:700; margin-bottom:6px;
}}
.price-big {{
  font-size:34px; font-weight:800; letter-spacing: -0.5px;
}}
.badge {{
  display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; font-size:12px;
  border:1px solid rgba(255,255,255,0.04);
}}
.badge.buy {{ background: linear-gradient(90deg,#064e3b,#10b981); color: #d1fae5; }}
.badge.sell {{ background: linear-gradient(90deg,#991b1b,#ef4444); color: #fee2e2; }}
.badge.neutral {{ background: rgba(71,85,105,0.18); color: #cbd5e1; }}

/* Buttons */
.button-exec {{
  width: 100%;
  background: linear-gradient(90deg,#2563eb,#4f46e5);
  color:white; padding:10px 14px; border-radius:10px; border:none; font-weight:700;
  cursor: pointer;
}}
.button-exec-disabled {{
  background: #334155;
  cursor: not-allowed;
}}
.button-ai {{
  background: rgba(30,41,59,0.6); color:#e2e8f0; padding:10px 14px; border-radius:10px; border:1px solid rgba(255,255,255,0.03);
  font-weight:700;
}}

/* Logs box */
.logs-box {{
  background: rgba(0,0,0,0.25);
  border-radius:10px; padding:12px; font-family: monospace; font-size:13px; color:#9fb6c8;
  border: 1px solid {GLASS_BORDER};
  max-height:220px; overflow:auto;
}}

/* small footers */
.small-muted {{ color: #94a3b8; font-size:13px; }}

/* Hide default Streamlit elements */
[data-testid="stHeader"], footer, .st-emotion-cache-19rxjzo {{ display: none !important; }}

</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ====== Assets configuration ======
ASSETS = {
    "PETR4": {"symbol":"PETR4","name":"PETR4","subtitle":"Petrobras PN","ponto":0.01,"dec":2, "tp_pontos":[62.5,75,100]},
    "WIN":   {"symbol":"WINZ25","name":"WIN","subtitle":"Mini Índice Futuro","ponto":0.2,"dec":0, "tp_pontos":[125,150,200]},
    "WDO":   {"symbol":"WDOZ25","name":"WDO","subtitle":"Mini Dólar Futuro","ponto":10,"dec":1, "tp_pontos":[25,30,40]}
}

# inicializa MT5 (tenta)
if not sinais.initialize_mt5():
    st.error("MT5 não conectado — verifique se o terminal está aberto e a conexão API está habilitada.", icon="🔌")

# Session state
if 'ordens' not in st.session_state:
    st.session_state.ordens = []

# autorefresh a cada 10s (ajuste ao seu gosto)
count = st_autorefresh(interval=10_000, key="refresh")

# ========== SIDEBAR ==========
st.sidebar.markdown("<div style='display:flex; gap:12px; align-items:center; margin-bottom:12px;'>"
                "<div style='width:44px;height:44px;border-radius:10px;background:linear-gradient(90deg,#6d28d9,#06b6d4);display:flex;align-items:center;justify-content:center;font-weight:800;color:white;'>IA</div>"
                "<div><div style='font-weight:800;font-size:18px'>IA finance</div><div class='small-muted'>Live Trading</div></div></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.radio("", ["Live Dashboard", "Research Lab", "Settings"], index=0, key="sidebar_nav")
st.sidebar.markdown("---")
st.sidebar.markdown("**User**")
st.sidebar.write("Marco Trader")
st.sidebar.caption("Pro Plan")
st.sidebar.markdown("---")
if st.sidebar.button("Toggle Theme", use_container_width=True):
    toggle_theme()
    st.rerun()

# ========== HEADER ==========
st.markdown(f"""
<h1 style="color:#e2e8f0; font-size:34px; font-weight:800;">
    <span style="color:{'#22c55e' if not sinais.sinal_bloqueado() else '#f87171'};">●</span> {page}
</h1>
<p style="margin-top:-10px; color:#94a3b8;">
    Connected to MetaTrader 5 (localhost) — Updated: {datetime.now().strftime('%H:%M:%S')}
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ====== Helper for mini area chart ======
def mini_area_chart(data_series, width=200, height=70, color_fill="#3b82f6"):
    if data_series is None or data_series.empty:
        return alt.Chart(pd.DataFrame({'x':[],'y':[]})).mark_area().properties(width=width, height=height)
    df = pd.DataFrame({"x": np.arange(len(data_series)), "y": data_series.values})
    chart = alt.Chart(df).mark_area(
        interpolate='monotone', opacity=0.18, color=color_fill
    ).encode(
        x=alt.X('x:Q', axis=None),
        y=alt.Y('y:Q', axis=None, scale=alt.Scale(zero=False))
    ).properties(width=width, height=height)
    line = alt.Chart(df).mark_line(interpolate='monotone', strokeWidth=2, color=color_fill).encode(x='x:Q', y='y:Q')
    return (chart + line).configure_view(strokeOpacity=0)

# ====== ROW OF CARDS ======
cols = st.columns(len(ASSETS))

for i, (k, cfg) in enumerate(ASSETS.items()):
    with cols[i]:
        disponível = sinais.selecionar_simbolo(cfg['symbol'])
        if not disponível:
            st.markdown(f"""
            <div class="card">
                <div class="asset-title">{cfg['name']}</div>
                <div style="font-size:16px; color:#ef4444; margin-top: 20px;">Símbolo indisponível no MT5.</div>
            </div>
            """, unsafe_allow_html=True)
            continue

        sinal, preco, confianca, sl_atr, tp_atr, atr_val, df_data = sinais.gerar_sinal_IA(cfg['symbol'], mt5.TIMEFRAME_M5)
        preco_display = f"{preco:,.{cfg['dec']}f}".replace(",", "X").replace(".", ",").replace("X", ".") if preco else "..."
        price_series = df_data['close'] if df_data is not None else pd.Series()

        # Create a form for each card to handle the button click
        with st.form(key=f"form_{cfg['symbol']}"):
            if sinal:
                sinal_tag = "buy"
                sinal_color = "#22c55e"
                if sinal == "VENDA":
                    sinal_tag = "sell"
                    sinal_color = "#ef4444"
                
                sl_em_reais = (sl_atr * atr_val) if atr_val and atr_val > 0 else 0
                sl_price = (preco - sl_em_reais) if sinal == "COMPRA" else (preco + sl_em_reais)
                tp_price = (preco + ((tp_atr * atr_val) if atr_val and atr_val > 0 else 0)) if sinal == "COMPRA" else (preco - ((tp_atr * atr_val) if atr_val and atr_val > 0 else 0))

                card_html = f'''
                <div class="card">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div class="asset-title">{cfg['name']}</div>
                            <div class="label-tag tag-{sinal_tag}">{sinal_tag.upper()}</div>
                        </div>
                        <div style="font-size:16px; color:#94a3b8;">{cfg['subtitle']}</div>
                        <div class="price-big">{preco_display}</div>
                        <div style="color:{sinal_color}; font-size:18px; font-weight:600;">
                            Confiança: {confianca:.1%}
                        </div>
                    </div>
                </div>
                '''
                st.markdown(card_html, unsafe_allow_html=True)
                
                st.altair_chart(mini_area_chart(price_series.tail(60), color_fill=sinal_color), use_container_width=True)

                st.markdown(f"""
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-top:8px; margin-bottom:12px;">
                  <div style="background: rgba(255,255,255,0.02); padding:8px; border-radius:8px;">
                    <div class='small-muted'>SL Price</div>
                    <div style="font-weight:800; font-size:18px; color:#ef4444;">{sl_price:,.{cfg['dec']}f}</div>
                  </div>
                  <div style="background: rgba(255,255,255,0.02); padding:8px; border-radius:8px;">
                    <div class='small-muted'>TP Price</div>
                    <div style="font-weight:800; font-size:18px; color:#22c55e;">{tp_price:,.{cfg['dec']}f}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                submitted = st.form_submit_button("EXECUTE")

                if submitted:
                    sl = sl_price
                    tps = []
                    for tp_mul in cfg['tp_pontos']:
                        current_tp_atr = tp_atr * (tp_mul / cfg['tp_pontos'][0]) if cfg['tp_pontos'] and cfg['tp_pontos'][0] > 0 else tp_atr
                        if sinal == "COMPRA": tps.append(preco + (current_tp_atr * atr_val))
                        else: tps.append(preco - (current_tp_atr * atr_val))
                    
                    resultados = []
                    sucesso_total = True
                    for idx,tp in enumerate(tps, start=1):
                        ok = sinais.executar_ordem(cfg['symbol'], sinal, preco, sl, tp, 1.0, idx)
                        resultados.append(f"TP{idx} ({round(tp,cfg['dec'])})→{'OK' if ok else 'Erro'}")
                        if not ok: sucesso_total = False
                    
                    st.session_state.ordens.append({
                        "time": time.strftime("%H:%M:%S"), "ativo": cfg['symbol'], "sinal": sinal,
                        "preco": preco_display, "resultados": ", ".join(resultados)
                    })
                    
                    try: winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                    except Exception: pass
                    
                    if sucesso_total: st.success("Ordem(ns) enviada(s) com sucesso!")
                    else: st.error("Falha ao enviar uma ou mais ordens.", icon="🚨")
            
            else: # No signal
                card_html = f"""
                <div class="card">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div class="asset-title">{cfg['name']}</div>
                            <div class="label-tag tag-neutral">NEUTRAL</div>
                        </div>
                        <div style="font-size:16px; color:#94a3b8;">{cfg['subtitle']}</div>
                        <div class="price-big">{preco_display}</div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                st.altair_chart(mini_area_chart(price_series.tail(60)), use_container_width=True)
                st.markdown("<div style='text-align:center; color:#94a3b8; margin-top:1rem;'>Aguardando sinal...</div>", unsafe_allow_html=True)
                st.markdown("<button class='button-exec button-exec-disabled' disabled>EXECUTE</button>", unsafe_allow_html=True)
                st.form_submit_button("Submit")

# ========== SYSTEM LOGS ==========
st.markdown("<br><h3 style='font-size:22px; font-weight:700;'>System Logs</h3>", unsafe_allow_html=True)
log_file_path = os.path.join(sinais.log_dir, 'ia_mt5.log')
log_content = "Log file not found."
if os.path.exists(log_file_path):
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            log_content = "".join(lines[-20:]).replace("<", "&lt;").replace(">", "&gt;")
    except Exception as e:
        log_content = f"Error reading log file: {e}"

st.markdown(f"<div class='logs-box'>{log_content}</div>", unsafe_allow_html=True)

# ========== RECENT ORDERS ==========
if st.session_state.ordens:
    st.markdown("<br><h3 style='font-size:22px; font-weight:700;'>Recent Orders</h3>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(st.session_state.ordens).iloc[::-1], use_container_width=True)

# chamada periódica para liberar lock se ordens fechadas
if sinais.sinal_bloqueado():
    fechado = sinais.verificar_ordens_fechadas()
    if fechado and not sinais.sinal_bloqueado():
        st.rerun()

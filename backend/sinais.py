# backend/sinais.py — IA 5min + NOTÍCIAS + ORDENS + JSON
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import psycopg2
import feedparser
import logging
import os
from datetime import datetime
from sklearn.linear_model import SGDClassifier
import filelock
import time
import json
from pathlib import Path

# === LOG ===
log_dir = 'C:\\temp'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('IA_MT5')
handler = logging.FileHandler(os.path.join(log_dir, 'ia_mt5.log'), encoding='utf-8')
handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(handler)

# === VARIÁVEIS GLOBAIS ===
ultimo_sinal = None
cache_local = {}
ultima_verificacao_noticias = None
cache_noticias = 0.0
historico_dados = {}
modelo = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, random_state=42)
melhores_params = {}  # ← armazenará os parâmetros da melhor estratégia

# === JSON DE RESULTADOS ===
ARQUIVO_RESULTADOS = Path("C:/Users/Marco/ia_finance/resultados.json")

def carregar_melhores_parametros_json(ativo):
    """
    Carrega os parâmetros do melhor backtest do JSON para o ativo especificado.
    """
    if not ARQUIVO_RESULTADOS.exists():
        logger.warning(f"Arquivo JSON de resultados não encontrado: {ARQUIVO_RESULTADOS}")
        return {}

    with open(ARQUIVO_RESULTADOS, "r", encoding="utf-8") as f:
        dados = json.load(f)

    resultados_ativo = [x for x in dados if x['ativo'] == ativo]
    if not resultados_ativo:
        return {}

    melhor = max(resultados_ativo, key=lambda x: x['lucro'])
    logger.info(f"Melhor backtest JSON para {ativo}: lucro={melhor['lucro']}, params={melhor['params']}")
    return melhor['params']

# === FUNÇÕES MT5 ===
def initialize_mt5(retries: int = 3, delay: float = 2.0):
    for tentativa in range(retries):
        if mt5.initialize():
            logger.info("✅ Conexão com MetaTrader 5 estabelecida.")
            return True
        else:
            logger.warning(f"Tentativa {tentativa+1}/{retries} falhou ao conectar ao MT5.")
            time.sleep(delay)
    raise RuntimeError("❌ Falha ao conectar ao MetaTrader 5 após múltiplas tentativas.")

def selecionar_simbolo(simbolo: str):
    if mt5.symbol_select(simbolo, True):
        logger.info(f"✅ Símbolo {simbolo} selecionado no MT5")
        return True
    logger.warning(f"⚠️ Símbolo {simbolo} não encontrado no MT5")
    return False

def executar_ordem(simbolo: str, tipo_ordem: str, preco: float, sl: float, tp: float, volume: float = 1.0, ordem_id: int = 1):
    try:
        if not mt5.initialize():
            logger.error("⚠️ MT5 não inicializado. Ordem não enviada.")
            return False

        tipo = mt5.ORDER_TYPE_BUY if tipo_ordem.upper() == "COMPRA" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": simbolo,
            "volume": volume,
            "type": tipo,
            "price": preco,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456 + ordem_id,
            "comment": f"IA_Parcial_{ordem_id}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ Falha ao executar ordem {tipo_ordem} para {simbolo}: {getattr(result, 'comment', 'Sem detalhe')}")
            return False

        logger.info(f"✅ Ordem {tipo_ordem} executada com sucesso para {simbolo} a {preco}")
        return True

    except Exception as e:
        logger.error(f"⚠️ Erro ao executar ordem {tipo_ordem} para {simbolo}: {e}")
        return False

# === INDICADORES ===
def calculate_rsi(series, period=14):
    delta = series.diff()
    if len(delta.dropna()) < period + 1:
        return np.nan
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).iloc[-1]

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd.iloc[-1], signal_line.iloc[-1]

def calculate_bollinger(series, window=20, stds=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma.iloc[-1] + (stds*std.iloc[-1]), sma.iloc[-1], sma.iloc[-1] - (stds*std.iloc[-1])

def calculate_ema(data, period):
    if len(data) < 5:
        return None, None
    ema_series = pd.Series(data).ewm(span=period, adjust=False).mean()
    return ema_series.iloc[-1], ema_series.iloc[-2]

# === MODELO ===
def treinar_modelo_IA(dados, simbolo):
    global historico_dados, modelo
    if simbolo not in historico_dados:
        historico_dados[simbolo] = pd.DataFrame(columns=['ema_curta','ema_longa','volatilidade','rsi','macd','bb_position','sinal'])
    dados = dados.dropna()
    if dados.empty:
        return None
    historico_dados[simbolo] = pd.concat([historico_dados[simbolo], dados], ignore_index=True)
    X = historico_dados[simbolo][['ema_curta','ema_longa','volatilidade','rsi','macd','bb_position']].values
    y = historico_dados[simbolo]['sinal'].values
    if len(X) < 2 or np.any(np.isnan(X)):
        return None
    modelo.partial_fit(X, y, classes=[-1,0,1])
    return modelo

# === FUNÇÕES DE CACHE / DB ===
def carregar_cache(simbolo):
    cache_path = os.path.join(log_dir, f"{simbolo}_cache.csv")
    with filelock.FileLock(f"{cache_path}.lock"):
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path)
    return pd.DataFrame(columns=['time','open','high','low','close','tick_volume','spread','real_volume'])

def salvar_cache_db(df, simbolo):
    cache_path = os.path.join(log_dir, f"{simbolo}_cache.csv")
    with filelock.FileLock(f"{cache_path}.lock"):
        df.to_csv(cache_path, index=False)

def salvar_sinal_db(ativo, sinal, preco, confianca):
    log_path = os.path.join(log_dir, "sinais_ao_vivo.csv")
    with filelock.FileLock(f"{log_path}.lock"):
        with open(log_path,"a",encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{ativo},{sinal},{preco},{confianca:.1f},OK\n")

# === NOTÍCIAS ===
def analisar_noticias():
    global ultima_verificacao_noticias, cache_noticias
    agora = datetime.now()
    if ultima_verificacao_noticias and (agora - ultima_verificacao_noticias).seconds < 600:
        return cache_noticias
    feeds = [
        'https://www.infomoney.com.br/feed/',
        'https://economia.uol.com.br/rss/',
        'https://valorinveste.globo.com/rss/ultimas.xml'
    ]
    palavras_alto_risco = ['copom','juros','inflação','recessão','crise','default','selic','fed']
    score_risco = 0.0
    for feed_url in feeds:
        try:
            rss = feedparser.parse(feed_url)
            for entry in rss.entries[:10]:
                titulo = (entry.title + " " + getattr(entry,'summary','')).lower()
                if any(p in titulo for p in palavras_alto_risco):
                    score_risco += 0.4
        except:
            continue
    score_risco = min(1.0, score_risco)
    cache_noticias = score_risco
    ultima_verificacao_noticias = agora
    return score_risco

# === IA PRINCIPAL ===
def gerar_sinal_IA(simbolo, timeframe, lookback=200):
    global ultimo_sinal
    # Carrega melhores parâmetros do JSON
    params_ativo = carregar_melhores_parametros_json(simbolo)
    if params_ativo:
        melhores_params.update(params_ativo)

    params = {
        "ema_curta": melhores_params.get("ema_curta", 9),
        "ema_longa": melhores_params.get("ema_longa", 21),
        "rsi_period": melhores_params.get("rsi_period", 14),
        "macd_fast": melhores_params.get("macd_fast", 8),
        "macd_slow": melhores_params.get("macd_slow", 17),
        "macd_signal": melhores_params.get("macd_signal", 9),
        "bb_window": melhores_params.get("bb_window", 20),
        "bb_std": melhores_params.get("bb_std", 2)
    }

    agora = datetime.now()
    hora_min = agora.hour*60 + agora.minute
    mercado_aberto = 600 <= hora_min <= 1050

    rates = mt5.copy_rates_from_pos(simbolo, timeframe, 0, lookback)
    if rates is None or len(rates) == 0:
        return None, None

    df = pd.DataFrame(rates)
    df['close'] = df['close'].astype(float)
    df['retorno'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)

    ema_curta_atual, ema_curta_anterior = calculate_ema(df['retorno'].dropna(), params["ema_curta"])
    ema_longa_atual, ema_longa_anterior = calculate_ema(df['retorno'].dropna(), params["ema_longa"])
    if ema_curta_atual is None or ema_longa_atual is None:
        return None, df['close'].iloc[-1]

    rsi = calculate_rsi(df['close'], params["rsi_period"])
    macd, signal_line = calculate_macd(df['close'], params["macd_fast"], params["macd_slow"], params["macd_signal"])
    upper, sma, lower = calculate_bollinger(df['close'], params["bb_window"], params["bb_std"])
    bb_position = (df['close'].iloc[-1] - sma) / (upper - lower) if (upper - lower) != 0 else 0

    cruzamento_compra = (ema_curta_atual > ema_longa_atual) and (ema_curta_anterior <= ema_longa_anterior)
    cruzamento_venda = (ema_curta_atual < ema_longa_atual) and (ema_curta_anterior >= ema_longa_anterior)

    volatilidade = df['retorno'].tail(20).std()
    ultimo_preco = df['close'].iloc[-1]

    dados_treino = pd.DataFrame({
        'ema_curta':[ema_curta_atual],
        'ema_longa':[ema_longa_atual],
        'volatilidade':[volatilidade],
        'rsi':[rsi],
        'macd':[macd],
        'bb_position':[bb_position],
        'sinal':[1 if cruzamento_compra else -1 if cruzamento_venda else 0]
    })

    modelo_local = treinar_modelo_IA(dados_treino, simbolo)
    confianca = abs(ema_curta_atual - ema_longa_atual)/ema_longa_atual if ema_longa_atual !=0 else 0
    risco_noticias = analisar_noticias()

    if risco_noticias > 0.5:
        return None, ultimo_preco

    if modelo_local is not None:
        previsao = modelo_local.predict([[ema_curta_atual, ema_longa_atual, volatilidade, rsi, macd, bb_position]])[0]
    else:
        previsao = 1 if cruzamento_compra else -1 if cruzamento_venda else 0

    if mercado_aberto and previsao==1 and cruzamento_compra and confianca>0.005:
        salvar_sinal_db(simbolo, "COMPRA", ultimo_preco, confianca)
        ultimo_sinal = "COMPRA"
        return "COMPRA", ultimo_preco
    elif mercado_aberto and previsao==-1 and cruzamento_venda and confianca>0.005:
        salvar_sinal_db(simbolo, "VENDA", ultimo_preco, confianca)
        ultimo_sinal = "VENDA"
        return "VENDA", ultimo_preco

    return None, ultimo_preco

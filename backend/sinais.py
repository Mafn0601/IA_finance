# backend/sinais.py — atualizado para carregar modelos por ativo (CSV-trained)
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
import joblib

# === CONFIG / PATHS ===
log_dir = 'C:\\temp'
os.makedirs(log_dir, exist_ok=True)
LOCK_PATH = os.path.join(log_dir, "sinal_lock.txt")
SINAIS_CSV = os.path.join(log_dir, "sinais_ao_vivo.csv")
ARQUIVO_RESULTADOS = Path("C:/Users/Marco/ia_finance/resultados.json")
MODELS_DIR = Path(__file__).resolve().parent / "modelos"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# === LOG ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('IA_MT5')
handler = logging.FileHandler(os.path.join(log_dir, 'ia_mt5.log'), encoding='utf-8')
handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
# avoid duplicate handlers if reloaded
if not logger.handlers:
    logger.addHandler(handler)
else:
    logger.handlers = [handler]

# === GLOBALS ===
ultimo_sinal = None
cache_local = {}
ultima_verificacao_noticias = None
cache_noticias = 0.0
historico_dados = {}
# modelo default para incremental (mantive, mas não mais usado para treino contínuo)
modelo_incremental = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, random_state=42)
melhores_params = {}

# ----------------------
# LOCK helpers
# ----------------------
def sinal_bloqueado():
    return os.path.exists(LOCK_PATH)

def bloquear_sinal():
    try:
        with open(LOCK_PATH, "w") as f:
            f.write(str(time.time()))
        logger.info("🔒 Lock criado: sinal bloqueado")
        return True
    except Exception as e:
        logger.error(f"Erro criar lock: {e}")
        return False

def liberar_sinal():
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
            logger.info("🔓 Lock removido: sinais liberados")
    except Exception as e:
        logger.error(f"Erro remover lock: {e}")

# ----------------------
# JSON loaders
# ----------------------
def carregar_melhores_parametros_json(ativo):
    if not ARQUIVO_RESULTADOS.exists():
        logger.debug(f"Arquivo JSON não existe: {ARQUIVO_RESULTADOS}")
        return {}
    try:
        with open(ARQUIVO_RESULTADOS, "r", encoding="utf-8") as f:
            dados = json.load(f)
    except Exception as e:
        logger.error(f"Erro lendo JSON {ARQUIVO_RESULTADOS}: {e}")
        return {}
    resultados_ativo = [x for x in dados if x.get('ativo') == ativo]
    if not resultados_ativo:
        return {}
    melhor = max(resultados_ativo, key=lambda x: x.get('lucro', -1))
    return melhor.get('params', {})

# ----------------------
# MODEL IO
# ----------------------
def model_path_for(symbol):
    return MODELS_DIR / f"{symbol}.pkl"

def carregar_modelo_IA(simbolo):
    """
    Carrega o modelo .pkl gerado pelo script de treino.
    Retorna o modelo (scikit-learn) ou None.
    """
    try:
        p = model_path_for(simbolo)
        if not p.exists():
            logger.info(f"Modelo não encontrado para {simbolo}: {p}")
            return None
        model = joblib.load(p)
        logger.info(f"Modelo carregado: {p}")
        return model
    except Exception as e:
        logger.error(f"Erro carregar_modelo_IA({simbolo}): {e}")
        return None

# ----------------------
# MT5 helpers
# ----------------------
def initialize_mt5(retries: int = 3, delay: float = 1.0):
    for tentativa in range(retries):
        try:
            ok = mt5.initialize()
        except Exception as e:
            logger.warning(f"mt5.initialize() exceção: {e}")
            ok = False
        if ok:
            logger.info("✅ MT5 inicializado")
            return True
        time.sleep(delay)
    logger.error("❌ Falha ao inicializar MT5")
    return False

def selecionar_simbolo(simbolo: str):
    try:
        ok = mt5.symbol_select(simbolo, True)
        if ok:
            logger.debug(f"Símbolo selecionado: {simbolo}")
            return True
        else:
            logger.warning(f"symbol_select falhou: {simbolo}")
            return False
    except Exception as e:
        logger.error(f"selecionar_simbolo erro: {e}")
        return False

def executar_ordem(symbol, sinal, preco, sl, tp, volume, tp_id):
    """Envio seguro de ordem (ajusta price, volume, digits, filling, deviation)."""
    try:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"executar_ordem: não selecionou {symbol}")
            return False

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"executar_ordem: info None para {symbol}")
            return False

        # volume ajustado
        try:
            vol_min = float(info.volume_min) if info.volume_min else 0.0
            vol_step = float(info.volume_step) if info.volume_step else 0.0
        except Exception:
            vol_min = 0.0; vol_step = 0.0

        if vol_min > 0 and volume < vol_min:
            volume = vol_min
        if vol_step > 0:
            n = round(volume / vol_step)
            volume = max(vol_min, n * vol_step)

        # price (ask for buy, bid for sell)
        if sinal.upper() == "COMPRA":
            price = float(info.ask) if hasattr(info, "ask") else float(preco)
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = float(info.bid) if hasattr(info, "bid") else float(preco)
            order_type = mt5.ORDER_TYPE_SELL

        digits = int(info.digits) if hasattr(info,'digits') and info.digits is not None else None
        if digits is not None:
            price = round(price, digits)
            sl = round(float(sl), digits)
            tp = round(float(tp), digits)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "type": order_type,
            "volume": float(volume),
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "magic": 20250125 + int(tp_id),
            "deviation": 20,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": getattr(mt5, "ORDER_FILLING_FOK", getattr(mt5, "ORDER_FILLING_IOC", mt5.ORDER_FILLING_IOC)),
            "comment": f"IA_TP{tp_id}"
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"executar_ordem: MT5 retornou None para {symbol}")
            return False

        logger.info(f"executar_ordem result: symbol={symbol} retcode={getattr(result,'retcode',None)} comment={getattr(result,'comment','')}")
        if hasattr(mt5, "TRADE_RETCODE_DONE") and getattr(result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
            return True

        logger.warning(f"executar_ordem rejeitada: retcode={getattr(result,'retcode',None)} comment={getattr(result,'comment','')}")
        return False
    except Exception as e:
        logger.error(f"executar_ordem exceção: {e}")
        return False

# ----------------------
# Indicadores / IA helpers
# ----------------------
def calculate_ema(data, period):
    if len(data) < period:
        return None, None
    s = pd.Series(data).ewm(span=period, adjust=False).mean()
    return s.iloc[-1], s.iloc[-2] if len(s) > 1 else s.iloc[-1]

def calculate_rsi(series, period=14):
    delta = series.diff()
    if len(delta.dropna()) < period + 1:
        return np.nan
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
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

def add_basic_indicators_df(df):
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    close = df["close"].astype(float)
    df["ema8"] = close.ewm(span=8, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["macd"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    tr = pd.concat([(df['high'] - df['low']).abs(), (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["ret"] = close.pct_change()
    df["vol20"] = df["ret"].rolling(20).std()
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df = df.dropna().reset_index(drop=True)
    return df

# ----------------------
# Cache / CSV
# ----------------------
def carregar_cache(simbolo):
    cache_path = os.path.join(log_dir, f"{simbolo}_cache.csv")
    with filelock.FileLock(f"{cache_path}.lock"):
        if os.path.exists(cache_path):
            try:
                return pd.read_csv(cache_path)
            except Exception as e:
                logger.error(f"carregar_cache: erro lendo {cache_path}: {e}")
                return pd.DataFrame()
    return pd.DataFrame()

def salvar_cache_db(df, simbolo):
    cache_path = os.path.join(log_dir, f"{simbolo}_cache.csv")
    with filelock.FileLock(f"{cache_path}.lock"):
        try:
            df.to_csv(cache_path, index=False)
        except Exception as e:
            logger.error(f"salvar_cache_db: erro escrevendo {cache_path}: {e}")

def salvar_sinal_db(ativo, sinal, preco, confianca):
    """Salva sinal (CSV) e cria lock para impedir novos sinais até ordem fechar."""
    try:
        # Se já existe lock, ignora
        if sinal_bloqueado():
            logger.info("salvar_sinal_db: lock presente -> ignorando gravação")
            return False

        # grava CSV com filelock
        with filelock.FileLock(f"{SINAIS_CSV}.lock"):
            new = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{ativo},{sinal},{preco},{confianca:.3f},OK\n"
            with open(SINAIS_CSV, "a", encoding="utf-8") as f:
                f.write(new)
        logger.info(f"Sinal salvo: {ativo} {sinal} {preco} conf={confianca:.3f}")

        # bloqueia novos sinais
        bloquear_sinal()
        return True
    except PermissionError as pe:
        logger.warning(f"PermissionError salvar_sinal_db: {pe} - tentando fallback no home")
        try:
            fallback = os.path.join(str(Path.home()), "ia_finance_sinais_ao_vivo.csv")
            with filelock.FileLock(f"{fallback}.lock"):
                with open(fallback, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{ativo},{sinal},{preco},{confianca:.3f},OK\n")
            bloquear_sinal()
            logger.info(f"Sinal salvo em fallback {fallback}")
            return True
        except Exception as e:
            logger.error(f"Falha fallback salvar_sinal_db: {e}")
            return False
    except Exception as e:
        logger.error(f"salvar_sinal_db erro: {e}")
        return False

# ----------------------
# NOTÍCIAS
# ----------------------
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
        except Exception as e:
            logger.debug(f"analisar_noticias: {e}")
            continue
    cache_noticias = min(1.0, score_risco)
    ultima_verificacao_noticias = agora
    return cache_noticias

# ----------------------
# gerar_sinal_IA (com modelo por ativo)
# ----------------------
def gerar_sinal_IA(simbolo, timeframe, lookback=200):
    global ultimo_sinal, melhores_params

    # não gerar se lock ativo
    if sinal_bloqueado():
        logger.debug("gerar_sinal_IA: lock ativo, não gera sinal")
        return None, None

    # atualiza params do JSON
    try:
        params_ativo = carregar_melhores_parametros_json(simbolo)
        if params_ativo:
            melhores_params.update(params_ativo)
    except Exception as e:
        logger.debug(f"erro carregar params json: {e}")

    # carrega modelo salvo (um por ativo)
    modelo = carregar_modelo_IA(simbolo)

    try:
        rates = mt5.copy_rates_from_pos(simbolo, timeframe, 0, lookback)
    except Exception as e:
        logger.error(f"gerar_sinal_IA: mt5.copy_rates_from_pos erro: {e}")
        return None, None

    if rates is None or len(rates) == 0:
        return None, None

    df = pd.DataFrame(rates)
    # converter time unix -> datetime
    if 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        # fallback
        df['datetime'] = pd.to_datetime(df.index, unit='s', errors='coerce')

    # garantir colunas
    for c in ['open','high','low','close']:
        if c not in df.columns:
            logger.error("coluna faltando em rates: " + c)
            return None, None
    df = df[['datetime','open','high','low','close']].copy()
    df = add_basic_indicators_df(df)
    if df is None or len(df) < 10:
        return None, None

    # extrair features coerentes com treino
    feat_cols = ["ema8","ema21","ema50","rsi14","macd","macd_signal","bb_width","vol20","stoch_k","stoch_d"]
    if not all(c in df.columns for c in feat_cols):
        # recalcula caso
        df = add_basic_indicators_df(df)

    latest = df.iloc[-1]
    X = np.array([latest[c] for c in feat_cols], dtype=float).reshape(1, -1)

    # confiança / prob thresholds
    prob_thresh = 0.55

    previsao = None
    confianca = 0.0
    if modelo is not None:
        try:
            probs = modelo.predict_proba(X)[0]
            # classes order depends on model.classes_
            classes = list(modelo.classes_)
            # choose max prob
            idx_max = int(np.argmax(probs))
            class_pred = classes[idx_max]
            confianca = float(probs[idx_max])
            logger.debug(f"modelo predict {class_pred} prob={confianca:.3f}")
            if confianca >= prob_thresh:
                previsao = int(class_pred)
            else:
                previsao = 0  # inconclusivo
        except Exception as e:
            logger.error(f"Erro predict modelo {simbolo}: {e}")
            previsao = 0
    else:
        # fallback por cruzamentos + heurística
        cruz_comp = (latest["ema8"] > latest["ema21"] and df["ema8"].iloc[-2] <= df["ema21"].iloc[-2])
        cruz_vend = (latest["ema8"] < latest["ema21"] and df["ema8"].iloc[-2] >= df["ema21"].iloc[-2])
        if cruz_comp:
            previsao = 1
            confianca = abs(latest["ema8"] - latest["ema21"]) / (abs(latest["ema21"]) + 1e-9)
        elif cruz_vend:
            previsao = -1
            confianca = abs(latest["ema8"] - latest["ema21"]) / (abs(latest["ema21"]) + 1e-9)
        else:
            previsao = 0
            confianca = 0.0

    # transformar previsao para sinal textual
    sinal = None
    if previsao == 1:
        sinal = "COMPRA"
    elif previsao == -1:
        sinal = "VENDA"
    else:
        sinal = None

    # bloqueios/evitar repetição
    if (ultimo_sinal == "COMPRA" and sinal == "COMPRA") or (ultimo_sinal == "VENDA" and sinal == "VENDA"):
        return None, float(df['close'].iloc[-1])

    # notícias
    risco_noticias = analisar_noticias()
    if risco_noticias > 0.5:
        logger.warning(f"Bloqueio por notícias ({risco_noticias:.2f})")
        return None, float(df['close'].iloc[-1])

    # só envia se mercado aberto (configurable)
    agora = datetime.now()
    hora_min = agora.hour*60 + agora.minute
    mercado_aberto = 600 <= hora_min <= 1050

    ultimo_preco = float(df['close'].iloc[-1])
    if mercado_aberto and sinal is not None and confianca > 0.005:
        ok = salvar_sinal_db(simbolo, sinal, ultimo_preco, confianca)
        if ok:
            ultimo_sinal = sinal
            return sinal, ultimo_preco

    return None, ultimo_preco

# ----------------------
# verificar ordens e liberar lock se nenhuma posição
# ----------------------
def verificar_ordens_fechadas():
    try:
        pos = mt5.positions_get()
        if pos is None or len(pos) == 0:
            # se não há posições abertas do robô, libera lock
            if sinal_bloqueado():
                liberar_sinal()
            return True
        # ainda tem posições -> manter lock
        return False
    except Exception as e:
        logger.debug(f"verificar_ordens_fechadas erro: {e}")
        return False

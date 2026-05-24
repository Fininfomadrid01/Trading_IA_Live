"""
GENERANDO CACHE DE PREDICCIONES (OPTIMIZADO)
=============================================
Carga datos 1H de España (Parquet) y Global (Yahoo Finance),
ejecuta el modelo CNN+BiLSTM+Attention y guarda el cache.

Optimizaciones:
- Lectura masiva de Parquet (pd.read_parquet sobre carpeta).
- Descarga directa 1h de Yahoo (máx 730 días).
- Paralelización interna de Pandas/Keras.
"""

import os, sys, warnings
os.environ["KERAS_BACKEND"]         = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import keras
from keras import ops
from pathlib import Path
from collections import defaultdict
import time

# --- CONFIGURACIÓN ---
BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
MC_PATH    = ROOT / "AJUSTADO DIARIO  1305" / "ibex35_parquet_dataset_adjusted_fixed"
MODEL_PATH = BASE_DIR / "Modelo_IA_Entrenado.keras"
CACHE_DIR  = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LEN  = 60
BATCH_SIZE  = 4096
MIN_PRECIO  = 0.1 # Bajamos un poco para incluir cryptos si es necesario

UNIVERSO_ESPAÑA = set([
    "A3M","ACS","ACX","ADX","AEDAS","AENA","AIR","ALB","ALM","AMS","ANA","ANE",
    "APPS","BBVA","BIO","BKIA","BKT","BME","CABK","CAF","CDR","CIE","CLNX",
    "COL","DOM","EBRO","ECR","EDR","EIDF","EKT","ELE","ENC","ENG","ENO","FAE",
    "FCC","FDR","FER","GCO","GEST","GRE","GRF","GSJ","HOME","IAG","IBE","IDR",
    "ITX","LDA","LLN","LOG","LRE","MAP","MAS","MCM","MEL","MRL","MTS","MVC",
    "NHH","NTGY","NTH","OHL","ORY","PHM","PRS","PSG","QBT","REE","REP","ROVI",
    "SAB","SAN","SCYR","SGRE","SLR","SOL","TEF","TL5","TLGO","TRE","TUB","UNI",
    "VID","VIS","ZOT"
])

UNIVERSO_GLOBAL = {
    "QQQ": "QQQ", "AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA", "TSLA": "TSLA",
    "GOLD": "GLD", "OIL": "USO", "SHY": "SHY", "FTSE": "^FTSE",
    "BTC": "BTC-USD", "ETH": "ETH-USD"
}

def descargar_datos_globales(ticker_yahoo):
    import yfinance as yf
    print(f"    Descargando {ticker_yahoo} (1h, 2y)...", end=" ", flush=True)
    try:
        # Yahoo permite hasta 730 días de 1h. Usamos period='2y' para maximizar.
        df = yf.download(ticker_yahoo, period="2y", interval="1h", progress=False, auto_adjust=True)
        if df.empty: 
            print("VACÍO")
            return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        print(f"OK ({len(df)} velas)")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return None

@keras.saving.register_keras_serializable()
def attention_block(x):
    score    = keras.layers.Dense(1, use_bias=False)(x)
    score    = keras.layers.Flatten()(score)
    alpha    = keras.layers.Softmax(name="attention_weights")(score)
    alpha_ex = keras.layers.Reshape((x.shape[1], 1))(alpha)
    context  = keras.layers.Multiply()([x, alpha_ex])
    context  = ops.sum(context, axis=1)
    return context, alpha

def build_all_windows(df_1h):
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(df_1h)
    if n < TARGET_LEN + 1: return np.zeros((0, TARGET_LEN, 8), np.float32)
    
    eps = np.float32(1e-9)
    o = df_1h["open"].values.astype(np.float32)
    h = df_1h["high"].values.astype(np.float32)
    l = df_1h["low"].values.astype(np.float32)
    c = df_1h["close"].values.astype(np.float32)
    v = df_1h["volume"].values.astype(np.float32)
    
    log_ret = np.zeros(n, np.float32)
    log_ret[1:] = np.diff(np.log(np.maximum(c, eps)))
    
    candle_rng = np.maximum(h - l, eps)
    per_bar = np.stack([
        log_ret,
        (h - l) / np.maximum(c, eps),
        (h - np.maximum(o, c)) / candle_rng,
        (np.minimum(o, c) - l) / candle_rng,
        np.abs(c - o) / candle_rng,
        np.concatenate([[0]*5, (c[5:]-c[:-5])/np.maximum(c[:-5],eps)])
    ], axis=1)
    
    nw = n - TARGET_LEN
    fw = sliding_window_view(per_bar, (TARGET_LEN, 6))[:nw, 0].copy()
    fw[:, 0, 0] = 0.0; fw[:, :5, 5] = 0.0
    
    vw = sliding_window_view(v, TARGET_LEN)[:nw]
    vr = np.clip(vw / np.maximum(vw.mean(1, keepdims=True), eps), 0, 10)
    
    cw = sliding_window_view(c, TARGET_LEN)[:nw]
    c_min = cw.min(1, keepdims=True)
    c_max = cw.max(1, keepdims=True)
    cn = (cw - c_min) / np.maximum(c_max - c_min, eps)
    
    return np.concatenate([fw, vr[:,:,None], cn[:,:,None]], axis=2).astype(np.float32)

def leer_año_completo_optimizado(año):
    año_path = MC_PATH / año
    if not año_path.exists(): return {}
    
    print(f"  Leyendo España {año} (archivo por archivo)...", end=" ", flush=True)
    chunks = defaultdict(list)
    try:
        files = sorted(año_path.glob("*.parquet"))
        for f in files:
            # Solo leer columnas necesarias para ahorrar memoria
            df = pd.read_parquet(f, columns=["VALOR", "HORA", "FECHA", "PRECIO", "VOLUMEN"])
            if "VALOR" not in df.columns: continue
            
            # Filtrar universo inmediatamente
            df = df[df["VALOR"].isin(UNIVERSO_ESPAÑA)]
            if df.empty: continue
            
            # Filtrar horas (9:00 a 17:30)
            hn = pd.to_numeric(df["HORA"], errors="coerce")
            mask = ((hn >= 9000000) & (hn <= 17300000)) | ((hn >= 90000000) & (hn <= 173000000))
            df = df[mask]
            if df.empty: continue
            
            for tk, g in df.groupby("VALOR"):
                chunks[tk].append(g)
        
        result = {}
        for tk, cs in chunks.items():
            d = pd.concat(cs, ignore_index=True)
            d["fecha_str"] = d["FECHA"].astype(str).str.zfill(8)
            hn_tk = pd.to_numeric(d["HORA"], errors="coerce")
            divisor = np.where(hn_tk > 9999999, 1000, 100)
            d["hora_str"] = (hn_tk // divisor).astype("Int64").astype(str).str.zfill(6)
            d["ts"] = pd.to_datetime(d["fecha_str"]+d["hora_str"], format="%Y%m%d%H%M%S", errors="coerce")
            d = d.dropna(subset=["ts"]).set_index("ts").sort_index()
            
            df1h = d.resample("1h").agg(open=("PRECIO","first"), high=("PRECIO","max"),
                                         low=("PRECIO","min"), close=("PRECIO","last"),
                                         volume=("VOLUMEN","sum")).dropna()
            if len(df1h) >= 10:
                result[tk] = df1h
        
        print(f"OK ({len(result)} tickers)")
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        return {}

def main():
    t_start = time.time()
    print("\nGENERANDO CACHE DE PREDICCIONES (MODO GLOBAL + ESPAÑA)")
    print("=" * 60)

    todos = defaultdict(list)

    # 1. Mercado Global
    print("\n[1/3] Procesando Mercado Global...")
    for tk_name, tk_yahoo in UNIVERSO_GLOBAL.items():
        df = descargar_datos_globales(tk_yahoo)
        if df is not None:
            todos[tk_name].append(df)

    # 2. Mercado España
    print("\n[2/3] Procesando Mercado España...")
    for año in ["2024", "2025"]:
        datos_año = leer_año_completo_optimizado(año)
        for tk, df in datos_año.items():
            todos[tk].append(df)

    # 3. Consolidar e Inferir
    print("\n[3/3] Consolidando e iniciando inferencia...")
    
    print("  Cargando modelo...", end=" ", flush=True)
    model = keras.models.load_model(str(MODEL_PATH))
    print("OK")

    tickers_finales = {}
    for tk, dfs in todos.items():
        df_full = pd.concat(dfs).sort_index()
        df_full = df_full[~df_full.index.duplicated(keep='first')]
        if len(df_full) >= TARGET_LEN + 1:
            tickers_finales[tk] = df_full

    tickers_lista = sorted(tickers_finales.keys())
    n_total = len(tickers_lista)
    
    cache_preds = {}
    cache_idx   = {}
    cache_c     = {}
    cache_l     = {}
    cache_h     = {}
    cache_v     = {}
    meta_rows   = []

    print(f"  Inferencia para {n_total} activos...")
    for i, tk in enumerate(tickers_lista):
        df_1h = tickers_finales[tk]
        X = build_all_windows(df_1h)
        
        if len(X) > 0:
            preds = model.predict(X, batch_size=BATCH_SIZE, verbose=0).astype(np.float16)
            
            # Volumen medio diario (sizing)
            vol_eur = (df_1h["close"] * df_1h["volume"]).resample("1D").sum()
            vol_medio_eur = float(vol_eur[vol_eur > 0].median()) if not vol_eur.empty else 0
            
            cache_preds[tk] = preds
            cache_idx[tk]   = df_1h.index.astype(np.int64).values
            cache_c[tk]     = df_1h["close"].values.astype(np.float32)
            cache_l[tk]     = df_1h["low"].values.astype(np.float32)
            cache_h[tk]     = df_1h["high"].values.astype(np.float32)
            cache_v[tk]     = df_1h["volume"].values.astype(np.float32)
            meta_rows.append({"ticker": tk, "n_barras": len(df_1h), "vol_medio_eur": round(vol_medio_eur, 0)})

        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t_start
            print(f"    Progreso: {i+1}/{n_total} | Ticker: {tk:<6} | Tiempo: {elapsed/60:.1f} min")

    # Guardar
    print(f"\n  Guardando cache en {CACHE_DIR}...", end=" ", flush=True)
    np.savez_compressed(
        str(CACHE_DIR / "cache_predicciones.npz"),
        **{f"preds_{tk}": v for tk, v in cache_preds.items()},
        **{f"idx_{tk}":   v for tk, v in cache_idx.items()},
        **{f"c_{tk}":     v for tk, v in cache_c.items()},
        **{f"l_{tk}":     v for tk, v in cache_l.items()},
        **{f"h_{tk}":     v for tk, v in cache_h.items()},
        **{f"v_{tk}":     v for tk, v in cache_v.items()},
    )
    pd.DataFrame(meta_rows).to_csv(CACHE_DIR / "cache_meta.csv", index=False)
    print("OK")

    print(f"\nPROCESO COMPLETADO EN {(time.time()-t_start)/60:.1f} MINUTOS")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario.")
    except Exception as e:
        print(f"\nERROR FATAL: {e}")
        import traceback
        traceback.print_exc()

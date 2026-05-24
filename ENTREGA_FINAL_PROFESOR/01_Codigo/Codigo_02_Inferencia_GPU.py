"""
GENERAR CACHE DE PREDICCIONES
=============================
Carga todos los datos 1H del Mercado Continuo (2018-2025),
ejecuta el modelo CNN+BiLSTM+Attention sobre cada ticker y
guarda las predicciones en disco (cache). Esto permite luego
probar distintas configuraciones de simulacion en segundos.

Salida: evaluacion_algoritmos/resultados/cache_predicciones/
  - cache_predicciones.npz  (predicciones + arrays de precios)
  - cache_meta.csv          (ticker, n_barras, vol_medio_eur)
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

ROOT       = Path("C:/Users/User/Desktop/VALIDAR HISTORICOS")
# MODIFIED TO USE FIXED DATASET
MC_PATH    = ROOT / "AJUSTADO DIARIO  1305" / "ibex35_parquet_dataset_adjusted_fixed"
MODEL_PATH = ROOT / "dataset_fase4_superclases" / "modelo_fase4_v2_best.keras"
CACHE_DIR  = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LEN  = 60
BATCH_SIZE  = 1024
MIN_PRECIO  = 1.0
TODOS_AÑOS  = ["2018","2019","2020","2021","2022","2023","2024","2025"]

UNIVERSO_LIMPIO = set([
    "A3M","ACS","ACX","ADX","AEDAS","AENA","AIR","ALB","ALM","AMS","ANA","ANE",
    "APPS","BBVA","BIO","BKIA","BKT","BME","CABK","CAF","CDR","CIE","CLNX",
    "COL","DOM","EBRO","ECR","EDR","EIDF","EKT","ELE","ENC","ENG","ENO","FAE",
    "FCC","FDR","FER","GCO","GEST","GRE","GRF","GSJ","HOME","IAG","IBE","IDR",
    "ITX","LDA","LLN","LOG","LRE","MAP","MAS","MCM","MEL","MRL","MTS","MVC",
    "NHH","NTGY","NTH","OHL","ORY","PHM","PRS","PSG","QBT","REE","REP","ROVI",
    "SAB","SAN","SCYR","SGRE","SLR","SOL","TEF","TL5","TLGO","TRE","TUB","UNI",
    "VID","VIS","ZOT"
])
UNIVERSO_LISTA = sorted(UNIVERSO_LIMPIO)

SUPER_LABELS = {0:"NONE",1:"ALC_CONT",2:"BAJ_CONT",3:"REV_ALCISTA",4:"REV_BAJISTA",5:"LATERAL"}

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
    eps = np.float32(1e-9)
    o = df_1h["open"].values.astype(np.float32)
    h = df_1h["high"].values.astype(np.float32)
    l = df_1h["low"].values.astype(np.float32)
    c = df_1h["close"].values.astype(np.float32)
    v = df_1h["volume"].values.astype(np.float32)
    log_ret = np.empty(n, np.float32); log_ret[0] = 0.0
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
    if nw <= 0:
        return np.zeros((0, TARGET_LEN, 8), np.float32)
    fw = sliding_window_view(per_bar, (TARGET_LEN, 6))[:nw, 0].copy()
    fw[:, 0, 0] = 0.0; fw[:, :5, 5] = 0.0
    vw = sliding_window_view(v, TARGET_LEN)[:nw]
    vr = np.clip(vw / np.maximum(vw.mean(1, keepdims=True), eps), 0, 10)
    cw = sliding_window_view(c, TARGET_LEN)[:nw]
    cn = (cw - cw.min(1,keepdims=True)) / np.maximum(cw.max(1,keepdims=True)-cw.min(1,keepdims=True), eps)
    return np.concatenate([fw, vr[:,:,None], cn[:,:,None]], axis=2).astype(np.float32)

def leer_año_completo(año):
    año_path = MC_PATH / año
    if not año_path.exists(): return {}
    chunks = defaultdict(list)
    for f in sorted(año_path.glob("*.parquet")):
        try:
            df = pd.read_parquet(f)
            if "VALOR" not in df.columns: continue
            df = df[df["VALOR"].isin(UNIVERSO_LIMPIO)]
            if df.empty: continue
            
            hc = pd.to_numeric(df["HORA"], errors="coerce")
            mask_7_8 = (hc >= 9000000) & (hc <= 17300000)
            mask_8_9 = (hc >= 90000000) & (hc <= 173000000)
            df = df[mask_7_8 | mask_8_9]
            if df.empty: continue
            
            for tk, g in df.groupby("VALOR"): chunks[tk].append(g)
        except: continue
        
    result = {}
    for tk, cs in chunks.items():
        d = pd.concat(cs, ignore_index=True)
        d["fecha_str"] = d["FECHA"].astype(str).str.zfill(8)
        hn = pd.to_numeric(d["HORA"], errors="coerce")
        
        # Divisor dinámico: si > 9,999,999 tiene formato largo (milisegundos)
        divisor = np.where(hn > 9999999, 1000, 100)
        
        d["hora_str"] = (hn // divisor).astype("Int64").astype(str).str.zfill(6)
        d["ts"] = pd.to_datetime(d["fecha_str"]+d["hora_str"], format="%Y%m%d%H%M%S", errors="coerce")
        d = d.dropna(subset=["ts"]).set_index("ts").sort_index()
        df1h = d.resample("1h").agg(open=("PRECIO","first"),high=("PRECIO","max"),
                                     low=("PRECIO","min"),close=("PRECIO","last"),
                                     volume=("VOLUMEN","sum")).dropna()
        if len(df1h) >= 10: result[tk] = df1h
    return result

def main():
    t0 = time.time()
    print("GENERANDO CACHE DE PREDICCIONES")
    print("=" * 60)

    print("  Cargando modelo...", end=" ", flush=True)
    model = keras.models.load_model(str(MODEL_PATH))
    print(f"OK ({time.time()-t0:.1f}s)")

    # Leer todos los años y consolidar por ticker
    print()
    todos = defaultdict(list)
    for año in TODOS_AÑOS:
        datos = leer_año_completo(año)
        print(f"  Leyendo {año}... {len(datos)} tickers")
        for tk, df in datos.items():
            todos[tk].append(df)

    tickers_consolidados = {}
    for tk, dfs in todos.items():
        df_full = pd.concat(dfs).sort_index()
        df_full = df_full[~df_full.index.duplicated(keep='first')]
        if df_full["close"].mean() >= MIN_PRECIO and len(df_full) >= TARGET_LEN + 10:
            tickers_consolidados[tk] = df_full

    tickers_lista = sorted(tickers_consolidados.keys())
    n_total = len(tickers_lista)
    print(f"\n  Consolidando... {n_total} tickers")
    print(f"\n  Generando predicciones y guardando cache...\n")

    # Calcular predicciones y guardar en NPZ
    cache_preds = {}  # ticker -> array float16 (n_windows, 6)
    cache_idx   = {}  # ticker -> array int64 (timestamps como ns)
    cache_c     = {}  # ticker -> close prices float32
    cache_l     = {}  # ticker -> low prices float32
    cache_h     = {}  # ticker -> high prices float32
    cache_v     = {}  # ticker -> volume float32
    meta_rows   = []

    for i, tk in enumerate(tickers_lista):
        pct = (i + 1) / n_total * 100
        df_1h = tickers_consolidados[tk]
        X = build_all_windows(df_1h)
        if len(X) == 0:
            continue

        # Prediccion vectorizada super rapida
        preds = model.predict(X, batch_size=4096, verbose=0).astype(np.float16)

        # Volumen medio diario en EUR (para sizing)
        vol_eur = (df_1h["close"] * df_1h["volume"]).resample("1D").sum()
        vol_medio_eur = float(vol_eur[vol_eur > 0].median())

        cache_preds[tk] = preds
        cache_idx[tk]   = df_1h.index.astype(np.int64).values
        cache_c[tk]     = df_1h["close"].values.astype(np.float32)
        cache_l[tk]     = df_1h["low"].values.astype(np.float32)
        cache_h[tk]     = df_1h["high"].values.astype(np.float32)
        cache_v[tk]     = df_1h["volume"].values.astype(np.float32)
        meta_rows.append({"ticker": tk, "n_barras": len(df_1h),
                          "vol_medio_eur": round(vol_medio_eur, 0)})

        if (i + 1) % 8 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_total - i - 1)
            print(f"  [{pct:5.1f}%] {tk:<6}  | {i+1}/{n_total} tickers "
                  f"| Elapsed {elapsed/60:.1f}min  ETA {eta/60:.1f}min")
            sys.stdout.flush()

    # Guardar cache
    print(f"\n  Guardando cache...", end=" ", flush=True)
    np.savez_compressed(
        str(CACHE_DIR / "cache_predicciones.npz"),
        **{f"preds_{tk}": v for tk, v in cache_preds.items()},
        **{f"idx_{tk}":   v for tk, v in cache_idx.items()},
        **{f"c_{tk}":     v for tk, v in cache_c.items()},
        **{f"l_{tk}":     v for tk, v in cache_l.items()},
        **{f"h_{tk}":     v for tk, v in cache_h.items()},
        **{f"v_{tk}":     v for tk, v in cache_v.items()},
    )
    df_meta = pd.DataFrame(meta_rows).sort_values("ticker")
    df_meta.to_csv(CACHE_DIR / "cache_meta.csv", index=False)
    n_ok = len(meta_rows)
    print(f"OK  ({n_ok} tickers guardados)")

    elapsed_total = (time.time() - t0) / 60
    print(f"\n  Cache guardada en: {CACHE_DIR}")
    print(f"  Tiempo total: {elapsed_total:.1f} minutos")
    print("\n  CACHE COMPLETADA - Ahora ejecuta optimizar_estrategia.py")

if __name__ == "__main__":
    main()

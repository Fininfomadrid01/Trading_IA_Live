import os
import numpy as np
import pandas as pd
import yfinance as yf
import keras
from pathlib import Path
import time
import warnings

# Silenciar warnings de TF/Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN ---
BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
CACHE_PATH = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones.npz"
MODEL_PATH = ROOT / "dataset_fase4_superclases" / "modelo_fase4_v2_best.keras"
OUT_CACHE  = ROOT / "evaluacion_algoritmos" / "resultados" / "cache_predicciones_fixed" / "cache_predicciones_LIVE.npz"

TARGET_LEN = 60

def get_yahoo_ticker(tk):
    special = {"BKIA": None, "BME": None, "MAS": None, "SGRE": None, "TL5": None, "REE": "REDEIA.MC"} 
    if tk in special: return special[tk]
    return f"{tk}.MC"

def build_windows_live(df_1h):
    n = len(df_1h)
    if n < TARGET_LEN + 1: return None
    
    eps = 1e-9
    o = df_1h["Open"].values.astype(np.float32)
    h = df_1h["High"].values.astype(np.float32)
    l = df_1h["Low"].values.astype(np.float32)
    c = df_1h["Close"].values.astype(np.float32)
    v = df_1h["Volume"].values.astype(np.float32)
    
    log_ret = np.zeros(n, np.float32)
    log_ret[1:] = np.diff(np.log(np.maximum(c, eps)))
    
    from numpy.lib.stride_tricks import sliding_window_view
    nw = n - TARGET_LEN
    
    feat_base = np.stack([
        log_ret,
        (h - l) / np.maximum(c, eps),
        (h - np.maximum(o, c)) / np.maximum(h - l, eps),
        (np.minimum(o, c) - l) / np.maximum(h - l, eps),
        np.abs(c - o) / np.maximum(h - l, eps),
        np.concatenate([[0]*5, (c[5:]-c[:-5])/np.maximum(c[:-5],eps)])
    ], axis=1)
    
    fw = sliding_window_view(feat_base, (TARGET_LEN, 6))[:nw, 0].copy()
    vw = sliding_window_view(v, TARGET_LEN)[:nw]
    vr = np.clip(vw / np.maximum(vw.mean(1, keepdims=True), eps), 0, 10)
    cw = sliding_window_view(c, TARGET_LEN)[:nw]
    cn = (cw - cw.min(1,keepdims=True)) / np.maximum(cw.max(1,keepdims=True)-cw.min(1,keepdims=True), eps)
    
    return np.concatenate([fw, vr[:,:,None], cn[:,:,None]], axis=2).astype(np.float32)

def main():
    print("SISTEMA DE ACTUALIZACIÓN DIARIA (YAHOO FINANCE LIVE)")
    print("="*60)
    
    if not CACHE_PATH.exists():
        print(f"ERROR: No se encuentra la cache en {CACHE_PATH}")
        return

    print(f"Cargando cache base...")
    old_data = np.load(str(CACHE_PATH))
    tickers = sorted([k.replace("preds_", "") for k in old_data.keys() if k.startswith("preds_")])
    
    print(f"Cargando modelo IA...")
    model = keras.models.load_model(str(MODEL_PATH))
    
    new_cache = {}
    t0 = time.time()

    for i, tk in enumerate(tickers):
        y_tk = get_yahoo_ticker(tk)
        if not y_tk: continue
        
        print(f"  [{i+1}/{len(tickers)}] {tk:<6}...", end=" ", flush=True)
        
        try:
            df_new = yf.download(y_tk, start="2026-01-01", interval="1h", progress=False, auto_adjust=True)
            if df_new.empty:
                print("Sin datos.")
                for pfx in ["preds", "idx", "c", "l", "h", "v"]:
                    new_cache[f"{pfx}_{tk}"] = old_data[f"{pfx}_{tk}"]
                continue
            
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)
            
            if df_new.index.tz is not None:
                df_new.index = df_new.index.tz_localize(None)

            c_old = old_data[f"c_{tk}"][-TARGET_LEN:]
            l_old = old_data[f"l_{tk}"][-TARGET_LEN:]
            h_old = old_data[f"h_{tk}"][-TARGET_LEN:]
            v_old = old_data[f"v_{tk}"][-TARGET_LEN:]
            idx_old = pd.to_datetime(old_data[f"idx_{tk}"][-TARGET_LEN:])
            
            if idx_old.tz is not None:
                idx_old = idx_old.tz_localize(None)
            
            df_old_context = pd.DataFrame({
                "Open": c_old, "High": h_old, "Low": l_old, "Close": c_old, "Volume": v_old
            }, index=idx_old)
            
            df_combined = pd.concat([df_old_context, df_new]).sort_index()
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
            
            X = build_windows_live(df_combined)
            if X is not None:
                new_preds_chunk = model.predict(X, batch_size=1024, verbose=0).astype(np.float16)
                
                new_cache[f"preds_{tk}"] = np.concatenate([old_data[f"preds_{tk}"], new_preds_chunk])
                new_cache[f"idx_{tk}"]   = np.concatenate([old_data[f"idx_{tk}"], df_new.index.astype(np.int64).values])
                new_cache[f"c_{tk}"]     = np.concatenate([old_data[f"c_{tk}"], df_new["Close"].values.astype(np.float32)])
                new_cache[f"l_{tk}"]     = np.concatenate([old_data[f"l_{tk}"], df_new["Low"].values.astype(np.float32)])
                new_cache[f"h_{tk}"]     = np.concatenate([old_data[f"h_{tk}"], df_new["High"].values.astype(np.float32)])
                new_cache[f"v_{tk}"]     = np.concatenate([old_data[f"v_{tk}"], df_new["Volume"].values.astype(np.float32)])
                print(f"OK (+{len(df_new)} velas)")
            else:
                print("Error ventanas.")
        except Exception as e:
            print(f"ERROR: {e}")
            for pfx in ["preds", "idx", "c", "l", "h", "v"]:
                new_cache[f"{pfx}_{tk}"] = old_data[f"{pfx}_{tk}"]

    print(f"\nGuardando cache LIVE...")
    np.savez_compressed(str(OUT_CACHE), **new_cache)
    print(f"¡PROCESO COMPLETADO en {(time.time()-t0)/60:.1f} min!")

if __name__ == "__main__":
    main()

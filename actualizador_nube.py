import os
os.environ["KERAS_BACKEND"] = "tensorflow"

# --- MENSAJE DE VERIFICACIÓN ---
print("\n" + "="*50)
print(">>> EJECUTANDO VERSIÓN CON PARCHE SAFEDENSE V2 <<<")
print("="*50 + "\n")

import numpy as np
import pandas as pd
import yfinance as yf
import keras
from pathlib import Path
import warnings

# === PARCHE SAFEDENSE ===
@keras.utils.register_keras_serializable(package="Custom")
class SafeDense(keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

CUSTOM_OBJECTS = {"Dense": SafeDense}
# ========================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

CACHE_PATH = Path("cache_predicciones_LIVE.npz")
MODEL_PATH = Path("modelo_fase4_v2_best.keras")
TARGET_LEN = 60

def get_yahoo_ticker(tk):
    return f"{tk}.MC" if tk != "REE" else "REDEIA.MC"

def build_windows_live(df):
    if len(df) < TARGET_LEN + 1: return None
    c = df["Close"].values.astype(np.float32)
    h = df["High"].values.astype(np.float32)
    l = df["Low"].values.astype(np.float32)
    o = df.get("Open", c).values.astype(np.float32)
    v = df["Volume"].values.astype(np.float32)
    lr = np.zeros(len(df), np.float32)
    lr[1:] = np.diff(np.log(np.maximum(c, 1e-9)))
    from numpy.lib.stride_tricks import sliding_window_view
    f = np.stack([lr, (h-l)/c, (h-np.maximum(o,c))/(h-l+1e-9), (np.minimum(o,c)-l)/(h-l+1e-9), np.abs(c-o)/(h-l+1e-9), np.concatenate([[0]*5, (c[5:]-c[:-5])/c[:-5]])], axis=1)
    fw = sliding_window_view(f, (TARGET_LEN, 6))[:-1, 0]
    vw = sliding_window_view(v, TARGET_LEN)[:-1]
    vr = np.clip(vw / np.maximum(vw.mean(1, keepdims=True), 1e-9), 0, 10)
    cw = sliding_window_view(c, TARGET_LEN)[:-1]
    cn = (cw - cw.min(1,keepdims=True)) / np.maximum(cw.max(1,keepdims=True)-cw.min(1,keepdims=True), 1e-9)
    return np.concatenate([fw, vr[:,:,None], cn[:,:,None]], axis=2).astype(np.float32)

def main():
    print("Iniciando carga de modelo...")
    if not CACHE_PATH.exists() or not MODEL_PATH.exists():
        print("Faltan archivos.")
        return
    
    old = np.load(str(CACHE_PATH))
    tickers = sorted([k.replace("preds_", "") for k in old.keys() if k.startswith("preds_")])
    
    # Esta línea ahora debería estar cerca de la 80
    model = keras.models.load_model(str(MODEL_PATH), custom_objects=CUSTOM_OBJECTS, compile=False)
    
    new_cache = {}
    for tk in tickers:
        try:
            df = yf.download(get_yahoo_ticker(tk), start="2026-01-01", interval="1h", progress=False, auto_adjust=True)
            if df.empty:
                for p in ["preds","idx","c","l","h","v"]: new_cache[f"{p}_{tk}"] = old[f"{p}_{tk}"]
                continue
            if df.index.tz: df.index = df.index.tz_localize(None)
            ctx = pd.DataFrame({"Close": old[f"c_{tk}"][-TARGET_LEN:], "High": old.get(f"h_{tk}", old[f"c_{tk}"])[-TARGET_LEN:], "Low": old.get(f"l_{tk}", old[f"c_{tk}"])[-TARGET_LEN:], "Volume": old[f"v_{tk}"][-TARGET_LEN:]})
            comb = pd.concat([ctx, df]).sort_index()
            comb = comb[~comb.index.duplicated(keep='last')]
            X = build_windows_live(comb)
            if X is not None:
                p = model.predict(X, verbose=0).astype(np.float16)
                new_cache[f"preds_{tk}"] = np.concatenate([old[f"preds_{tk}"], p])
                new_cache[f"idx_{tk}"] = np.concatenate([old[f"idx_{tk}"], df.index.astype(np.int64).values])
                new_cache[f"c_{tk}"] = np.concatenate([old[f"c_{tk}"], df["Close"].values])
                new_cache[f"v_{tk}"] = np.concatenate([old[f"v_{tk}"], df["Volume"].values])
                new_cache[f"l_{tk}"] = np.concatenate([old[f"l_{tk}"], df.get("Low", df["Close"]).values])
                new_cache[f"h_{tk}"] = np.concatenate([old[f"h_{tk}"], df.get("High", df["Close"]).values])
                print(f" > {tk} OK")
        except Exception as e:
            print(f" > {tk} ERROR: {e}")
            for p in ["preds","idx","c","l","h","v"]: new_cache[f"{p}_{tk}"] = old[f"{p}_{tk}"]
    np.savez_compressed(str(CACHE_PATH), **new_cache)
    print("TERMINADO.")

if __name__ == "__main__": main()


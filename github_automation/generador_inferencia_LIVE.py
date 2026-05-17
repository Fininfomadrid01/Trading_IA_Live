import os, sys, warnings
os.environ["KERAS_BACKEND"] = "tensorflow"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import keras
from pathlib import Path
from datetime import datetime

# --- CONFIGURACIÓN DE RUTAS ---
ROOT = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
MODEL_PATH = ROOT / "Modelo_IA_Entrenado.keras"
CACHE_PATH = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\evaluacion_algoritmos\resultados\cache_predicciones_fixed\cache_predicciones_LIVE.npz")

MARKET_PATHS = {
    'IBEX':   ROOT.parent / "AJUSTADO DIARIO  1305" / "ibex35_parquet_dataset_adjusted_fixed" / "2026",
    'CRYPTO': ROOT / "EXPERIMENTO_CRYPTO" / "datos_raw",
    'COMMO':  ROOT / "EXPERIMENTO_COMMODITIES" / "datos_raw",
    'NASDAQ': ROOT / "EXPERIMENTO_NASDAQ" / "datos_raw"
}

TARGET_LEN = 60

def build_windows(df):
    # Simplificación de la lógica de ventanas para inferencia rápida
    c = df["close"].values.astype(np.float32)
    v = df["volume"].values.astype(np.float32)
    log_ret = np.zeros_like(c)
    log_ret[1:] = np.diff(np.log(np.maximum(c, 1e-9)))
    
    # Fake features for the 8-dim structure (maintaining compatibility)
    f2 = np.zeros_like(c) 
    f3 = np.zeros_like(c)
    f4 = np.zeros_like(c)
    f5 = np.zeros_like(c)
    f6 = np.zeros_like(c)
    
    per_bar = np.stack([log_ret, f2, f3, f4, f5, f6], axis=1)
    
    X = []
    for i in range(len(df) - TARGET_LEN):
        window_feat = per_bar[i:i+TARGET_LEN]
        window_vol  = v[i:i+TARGET_LEN]
        vol_norm    = window_vol / (window_vol.mean() + 1e-9)
        window_price = c[i:i+TARGET_LEN]
        price_norm  = (window_price - window_price.min()) / (window_price.max() - window_price.min() + 1e-9)
        
        combined = np.concatenate([window_feat, vol_norm[:,None], price_norm[:,None]], axis=1)
        X.append(combined)
    return np.array(X).astype(np.float32)

def main():
    print("INICIANDO INFERENCIA GLOBAL MULTIACTIVO")
    model = keras.models.load_model(str(MODEL_PATH))
    
    final_cache = {}
    
    for market, path in MARKET_PATHS.items():
        print(f"\n--- Procesando Mercado: {market} ---")
        if not path.exists():
            print(f"Ruta no encontrada: {path}")
            continue
            
        files = list(path.glob("*.parquet"))
        for f in files:
            ticker = f.stem.split("_")[0]
            try:
                df = pd.read_parquet(f)
                if df.empty: continue
                
                # Normalizar columnas (maneja strings y tuplas de MultiIndex)
                df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
                
                if 'close' not in df.columns or len(df) < TARGET_LEN + 5: continue
                
                X = build_windows(df)
                if len(X) == 0: continue
                
                preds = model.predict(X, verbose=0).astype(np.float16)
                
                final_cache[f"preds_{ticker}"] = preds
                final_cache[f"idx_{ticker}"]   = df.index.astype(np.int64).values
                final_cache[f"c_{ticker}"]     = df["close"].values.astype(np.float32)
                
                print(f"OK: {ticker} procesado ({len(preds)} ventanas)")
            except Exception as e:
                print(f"Error en {ticker}: {e}")

    print(f"\nGuardando Cache Global en {CACHE_PATH}")
    np.savez_compressed(str(CACHE_PATH), **final_cache)
    print("Proceso completado exitosamente.")

if __name__ == "__main__":
    main()

import yfinance as yf
import pandas as pd
from pathlib import Path

# --- CONFIGURACIÓN ---
BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
COMM_DIR = BASE_DIR / "EXPERIMENTO_COMMODITIES" / "datos_raw"
COMM_DIR.mkdir(parents=True, exist_ok=True)

# GC=F (Gold Futures), CL=F (Crude Oil Futures)
TICKERS = ["GC=F", "CL=F"]

def descargar_commodities():
    print(f"Descargando datos de Materias Primas en {COMM_DIR}...")
    for t in TICKERS:
        print(f" > {t}...")
        # 1H para inferencia
        df_1h = yf.download(t, period="2y", interval="1h", progress=False, auto_adjust=True)
        if not df_1h.empty:
            name = "GOLD" if "GC" in t else "OIL"
            df_1h.to_parquet(COMM_DIR / f"{name}_1H.parquet")
        
        # 1D para benchmark
        df_1d = yf.download(t, period="max", interval="1d", progress=False, auto_adjust=True)
        if not df_1d.empty:
            name = "GOLD" if "GC" in t else "OIL"
            df_1d.to_parquet(COMM_DIR / f"{name}_1D.parquet")

if __name__ == "__main__":
    descargar_commodities()
    print("\nMaterias primas listas para el laboratorio.")

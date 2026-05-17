import yfinance as yf
import pandas as pd
from pathlib import Path
import os

# --- CONFIGURACIÓN ---
BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
CRYPTO_DIR = BASE_DIR / "EXPERIMENTO_CRYPTO" / "datos_raw"
CRYPTO_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]

def descargar_crypto():
    print(f"Descargando datos del Top 5 Crypto en {CRYPTO_DIR}...")
    for t in TICKERS:
        print(f" > {t}...")
        # Descargar 1H (últimos 2 años es el máximo para 1h en yfinance)
        df_1h = yf.download(t, period="2y", interval="1h", progress=False, auto_adjust=True)
        if not df_1h.empty:
            df_1h.to_parquet(CRYPTO_DIR / f"{t}_1H.parquet")
        
        # Descargar 1D (para el filtro de régimen/benchmark)
        df_1d = yf.download(t, period="max", interval="1d", progress=False, auto_adjust=True)
        if not df_1d.empty:
            df_1d.to_parquet(CRYPTO_DIR / f"{t}_1D.parquet")

if __name__ == "__main__":
    descargar_crypto()
    print("\nDescarga completada. Listos para la validación cruzada.")

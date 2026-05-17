import yfinance as yf
import pandas as pd
from pathlib import Path

OUT_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\EXPERIMENTO_NASDAQ\datos_raw")
# Wait, I should put it in SP500 folder
OUT_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL\EXPERIMENTO_SP500\datos_raw")

def download_spy():
    print("Descargando SPY (S&P 500 Benchmark)...")
    data = yf.download("SPY", start="2018-01-01", interval="1d", progress=False)
    if not data.empty:
        data.to_parquet(OUT_DIR / "SPY_1D.parquet")
        print("SPY guardado con éxito.")

if __name__ == "__main__":
    download_spy()

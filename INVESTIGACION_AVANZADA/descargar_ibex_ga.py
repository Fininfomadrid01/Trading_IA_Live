import yfinance as yf
from pathlib import Path

# Carpeta de datos para el GA del IBEX
BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
IBEX_GA_DIR = BASE_DIR / "INVESTIGACION_AVANZADA" / "datos_ibex"
IBEX_GA_DIR.mkdir(parents=True, exist_ok=True)

def download_sample_ibex():
    tickers = ["SAN.MC", "TEF.MC", "BBVA.MC", "ITX.MC", "REP.MC", "^IBEX"]
    print(f"Descargando {len(tickers)} activos del IBEX 35...")
    for t in tickers:
        print(f" > {t}")
        # 1H (730d)
        data = yf.download(t, period="730d", interval="1h", progress=False)
        if not data.empty:
            data.to_parquet(IBEX_GA_DIR / f"{t}_1H.parquet")
        # 1D
        data_d = yf.download(t, period="max", interval="1d", progress=False)
        if not data_d.empty:
            data_d.to_parquet(IBEX_GA_DIR / f"{t}_1D.parquet")

if __name__ == "__main__":
    download_sample_ibex()

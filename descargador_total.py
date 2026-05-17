import yfinance as yf
import pandas as pd
import requests
import os
from pathlib import Path

BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
headers = {'User-Agent': 'Mozilla/5.0'}

def get_tickers(url, table_idx, col_name, suffix=""):
    try:
        response = requests.get(url, headers=headers)
        tables = pd.read_html(response.text)
        df = tables[table_idx]
        return [f"{t}{suffix}" for t in df[col_name].tolist()]
    except Exception as e:
        print(f"Error en {url}: {e}")
        return []

def download_market(name, tickers, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Descargando {name} ({len(tickers)} activos) ---")
    for i, t in enumerate(tickers):
        # Limpieza de ticker (algunos traen puntos o guiones raros)
        t = str(t).replace(".", "-") if name == "S&P 500" else t
        t = t.split(" ")[0] # Para evitar espacios
        
        # 1H
        f1h = out_dir / f"{t}_1H.parquet"
        if not f1h.exists():
            try:
                print(f"[{i+1}/{len(tickers)}] {t} (1H)", end="\r")
                data = yf.download(t, period="730d", interval="1h", progress=False)
                if not data.empty: data.to_parquet(f1h)
            except: pass
        
        # 1D
        f1d = out_dir / f"{t}_1D.parquet"
        if not f1d.exists():
            try:
                print(f"[{i+1}/{len(tickers)}] {t} (1D)", end="\r")
                data = yf.download(t, start="2018-01-01", interval="1d", progress=False)
                if not data.empty: data.to_parquet(f1d)
            except: pass

if __name__ == "__main__":
    # 1. IBEX 35
    ibex_tickers = get_tickers('https://en.wikipedia.org/wiki/IBEX_35', 2, 'Ticker', "")
    download_market("IBEX 35", ibex_tickers, BASE_DIR / "INVESTIGACION_AVANZADA" / "datos_ibex")
    
    # 2. NASDAQ 100
    nasdaq_tickers = get_tickers('https://en.wikipedia.org/wiki/Nasdaq-100', 4, 'Ticker')
    download_market("NASDAQ 100", nasdaq_tickers, BASE_DIR / "EXPERIMENTO_NASDAQ" / "datos_raw")
    
    # 3. FTSE 100
    ftse_tickers = get_tickers('https://en.wikipedia.org/wiki/FTSE_100_Index', 6, 'Ticker', ".L")
    download_market("FTSE 100", ftse_tickers, BASE_DIR / "EXPERIMENTO_FTSE100" / "datos_raw")
    
    # 4. S&P 500
    sp_tickers = get_tickers('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 0, 'Symbol')
    download_market("S&P 500", sp_tickers, BASE_DIR / "EXPERIMENTO_SP500" / "datos_raw")

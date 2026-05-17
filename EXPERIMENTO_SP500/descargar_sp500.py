import yfinance as yf
import pandas as pd
import os
from pathlib import Path

# Configuración
BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
OUT_DIR = BASE_DIR / "EXPERIMENTO_SP500" / "datos_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def download_sp500_data():
    print("Obteniendo lista de componentes del S&P 500 desde Wikipedia...")
    try:
        import requests
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url, headers=headers)
        table = pd.read_html(response.text)
        df = table[0]
        tickers = df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
    except Exception as e:
        print(f"Error obteniendo tickers: {e}")
        return

    print(f"Iniciando descarga de {len(tickers)} activos del S&P 500...")
    
    # Bajamos el SPY como referencia de régimen
    tickers.append("SPY")
    
    for i, symbol in enumerate(tickers):
        try:
            print(f"[{i+1}/{len(tickers)}] Descargando {symbol}...", end="\r")
            # 1H (730d)
            if not (OUT_DIR / f"{symbol}_1H.parquet").exists():
                data = yf.download(symbol, period="730d", interval="1h", progress=False)
                if not data.empty:
                    data.to_parquet(OUT_DIR / f"{symbol}_1H.parquet")
            
            # 1D (2018)
            if not (OUT_DIR / f"{symbol}_1D.parquet").exists():
                data_daily = yf.download(symbol, start="2018-01-01", interval="1d", progress=False)
                if not data_daily.empty:
                    data_daily.to_parquet(OUT_DIR / f"{symbol}_1D.parquet")
                
        except Exception as e:
            print(f"\nError con {symbol}: {e}")

if __name__ == "__main__":
    download_sp500_data()

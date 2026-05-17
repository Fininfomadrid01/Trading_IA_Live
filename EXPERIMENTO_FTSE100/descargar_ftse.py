import yfinance as yf
import pandas as pd
import requests
from pathlib import Path

# Configuración
BASE_DIR = Path(r"C:\Users\User\Desktop\VALIDAR HISTORICOS\ENTREGA_TFM_VERSION_FINAL")
OUT_DIR = BASE_DIR / "EXPERIMENTO_FTSE100" / "datos_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def download_ftse100_data():
    print("Obteniendo lista de componentes del FTSE 100 desde Wikipedia...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = 'https://en.wikipedia.org/wiki/FTSE_100_Index'
        response = requests.get(url, headers=headers)
        table = pd.read_html(response.text)
        df = table[6] # Índice 6 para los constituyentes
        tickers = [f"{t}.L" for t in df['Ticker'].tolist()]
    except Exception as e:
        print(f"Error obteniendo tickers: {e}")
        # Backup list en caso de fallo de Wikipedia
        tickers = ["BP.L", "HSBA.L", "SHEL.L", "GSK.L", "AZN.L", "ULVR.L", "DGE.L", "RIO.L", "LLOY.L", "BARC.L"]
        print("Usando lista de backup (principales activos).")

    print(f"Iniciando descarga de {len(tickers)} activos del FTSE 100...")
    
    # Bajamos el ^FTSE como referencia
    tickers.append("^FTSE")
    
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
    download_ftse100_data()
